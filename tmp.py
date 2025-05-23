#!/usr/bin/env python3
# ===================================================================
#  EZ-GNN end-to-end (trainable temporal encoder + attention pool)
#  ────────────────────────────────────────────────────────────────
#  • Raw 2048 Hz baseline, 300 s
#  • Node feature  = quantised xyz  (N,3)
#  • Edge feature  = z_edge learned on-the-fly (E,64)
#  • GPU budget    ≈ 6 GB  → edges processed in small chunks
#  -----------------------------------------------------------------
#  Edit the PATHS + PATIENT ID lists at the bottom and run:
#      python graph.py
# ===================================================================

"""
Key patches 2025‑05‑21
─────────────────────
1.  **Loss**  ⟶ weighted BCE‑with‑logits (`pos_weight=7.0`).
    • `FocalBCELoss` is kept for experiments but is **not used** – see   
      commented‑out block in `__main__`.
2.  **Model head** now returns *raw logits*; `sigmoid` removed.  
   (Metrics & loss handle logits directly.)
3.  **Grad‑clip** is applied *after* `scaler.unscale_()`.
4.  Other cosmetic clean‑ups only.
"""

import os, time, warnings, numpy as np, pandas as pd, torch, mne
from itertools import combinations
from pathlib import Path
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchmetrics
import wandb

from torch.amp import autocast, GradScaler

# ---------------- 0.  HOUSE‑KEEPING -------------------------
scaler = GradScaler("cuda")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.backends.cudnn.benchmark = True

# ---------------- 1.  GLOBAL CONSTANTS ----------------------
FS_HZ        = 2048               # sampling rate
WIN_SEC      = .5
STRIDE_SEC   = 2
WIN_LEN      = int(WIN_SEC   * FS_HZ)
STRIDE       = int(STRIDE_SEC * FS_HZ)
Z_DIM        = 16
NODE_DIM     = 3
HID_DIM      = 8 # Updated from 16
LAYERS       = 3 # Updated from 4
BASE_EDGE_CHUNK = 32
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- 2.  TEMPORAL ENCODER ----------------------
class TemporalEncoder(torch.nn.Module):
    """Dilated TCN: (B,1,L) → (B, Z_DIM)"""
    def __init__(self, hid=16, out_dim=Z_DIM, k=5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(1, hid, k, dilation=1, padding=2),
            torch.nn.Conv1d(hid, hid, k, dilation=2, padding=4),
            torch.nn.Conv1d(hid, hid, k, dilation=4, padding=8),
            torch.nn.Conv1d(hid, out_dim, k, dilation=8, padding=16)
        ])
        self.act = torch.nn.ReLU()

    def forward(self, x):                     # (B,1,WIN_LEN)
        for c in self.convs:
            x = self.act(c(x))
        return torch.amax(x, dim=-1)          # (B,Z_DIM)

# ---------------- 3.  EDGE‑LEVEL POOL -----------------------
class AttentionPool(torch.nn.Module):
    def __init__(self, dim=Z_DIM):
        super().__init__(); self.q = torch.nn.Parameter(torch.randn(dim))
    def forward(self, z_win):                 # (E, W, Z_DIM)
        α = torch.softmax(torch.einsum('ewd,d->ew', z_win, self.q), dim=1)
        return torch.einsum('ewd,ew->ed', z_win, α)        # (E, Z_DIM)

# ---------------- 4.  GNN MESSAGE PASSING -------------------
class EdgeMPNN(MessagePassing):
    def __init__(self, h):
        super().__init__(aggr='add')
        self.lin_i = torch.nn.Linear(h, h)
        self.lin_j = torch.nn.Linear(h, h)
        self.lin_e = torch.nn.Linear(h, h)
        self.mlp   = torch.nn.Sequential(
            torch.nn.Linear(3*h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.LayerNorm(h)
        )
        self.act = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([self.lin_i(x_i), self.lin_j(x_j), self.lin_e(edge_attr)], -1)
        return self.mlp(m)

    def update(self, aggr_out, x):
        return self.act(x + aggr_out)

# ---------------- 5.  FULL NETWORK --------------------------
class EZGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal   = TemporalEncoder()
        self.tpool      = AttentionPool()
        self.node_proj  = torch.nn.Linear(NODE_DIM, HID_DIM)
        self.edge_proj  = torch.nn.Linear(Z_DIM,   HID_DIM)
        self.gnn_layers = torch.nn.ModuleList([EdgeMPNN(HID_DIM) for _ in range(LAYERS)])
        # Added dropout layers as per minimal recipe
        self.dropout_layers = torch.nn.ModuleList([torch.nn.Dropout(0.25) for _ in range(LAYERS)])
        self.out_head   = torch.nn.Sequential(
            torch.nn.LayerNorm(HID_DIM),
            torch.nn.Linear(HID_DIM, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def _compute_edge_attr(self, signals, edge_index):
        E = edge_index.size(1)
        N = signals.size(0)
        edge_chunk = int(max(16, BASE_EDGE_CHUNK * (100/max(N,1)))) if N > 150 else BASE_EDGE_CHUNK
        edge_attr = []
        for s in range(0, E, edge_chunk):
            idx  = edge_index[:, s:s+edge_chunk]
            diff = signals[idx[0]] - signals[idx[1]]
            win  = diff.unfold(-1, WIN_LEN, STRIDE)
            B, W, L = win.shape
            win_reshaped = win.reshape(-1,1,L).contiguous().requires_grad_(True)
            z_win = checkpoint(self.temporal, win_reshaped, use_reentrant=False)
            z_edge = self.tpool(z_win.view(B, W, Z_DIM))
            edge_attr.append(z_edge)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(edge_attr, 0)

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr_computed = self._compute_edge_attr(data.signals, data.edge_index)
        edge_attr = self.edge_proj(edge_attr_computed)
        for i, gnn in enumerate(self.gnn_layers):
            x = gnn(x, data.edge_index, edge_attr)
            x = self.dropout_layers[i](x) # Apply dropout after each GNN layer
        logits = self.out_head(x).view(-1)
        return logits

# ---------------- 6.  LOSSES --------------------------------
class FocalBCELoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, target):
        p  = torch.sigmoid(logits)
        pt = torch.where(target==1, p, 1-p).clamp(1e-6, 1-1e-6)
        w  = self.a*target + (1-self.a)*(1-target)
        return ( - w * (1-pt)**self.g * torch.log(pt) ).mean()

# ---------------- 7.  GRAPH BUILDER -------------------------
def build_patient_graph(pid: int, edf_dir: str, contact_csv: str):
    contact_df = pd.read_csv(contact_csv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(os.path.join(edf_dir, f"{pid:03}_Baseline.EDF"),
                                  preload=True, verbose=False)
    assert int(raw.info['sfreq']) == FS_HZ
    sf = raw.info['sfreq']
    pat = contact_df[contact_df.pid == pid].copy()
    raw_ch_no_prefix = [ch.replace('EEG ', '').strip() for ch in raw.ch_names if 'EEG ' in ch]
    pat['eeg_match'] = pat.electrode.str.strip()
    valid = pat[pat.eeg_match.isin(raw_ch_no_prefix)]
    picks_map = {row.eeg_match: f"EEG {row.electrode.strip()}" for _, row in valid.iterrows()}
    ordered_picks = [picks_map[n] for n in valid.eeg_match]
    raw.notch_filter(np.arange(60, int(sf//2), 60), picks=ordered_picks,
                     filter_length='auto', phase='zero',
                     fir_window='hamming', fir_design='firwin', verbose=False)
    data = raw.get_data(picks=ordered_picks).astype(np.float32)
    data = data[:, :int(sf*10)] * 1e-6
    xyz = valid[['X_recoded_dl2','Y_recoded_dl2','Z_recoded_dl2']].to_numpy(float)
    x_node = torch.from_numpy(xyz).float()
    N = x_node.size(0)
    pairs = list(combinations(range(N), 2))
    src, dst = zip(*pairs)
    edge_idx = torch.tensor(np.vstack([src+dst, dst+src]), dtype=torch.long)
    y_node = torch.tensor(valid.soz.values, dtype=torch.float32)
    signals = torch.from_numpy(data)
    return Data(x=x_node, edge_index=edge_idx, y_node=y_node, signals=signals, pid=torch.tensor([pid]))

# ---------------- 8.  DATASET -------------------------------
class EZDataset(torch.utils.data.Dataset):
    def __init__(self, pids, edf_dir, contact_csv):
        self.pids, self.edf_dir, self.contact_csv = pids, edf_dir, contact_csv
    def __len__(self):  return len(self.pids)
    def __getitem__(self, idx):
        return build_patient_graph(self.pids[idx], self.edf_dir, self.contact_csv)

# ---------------- 9.  TRAIN / EVAL LOOPS --------------------
def save_checkpoint(model, optimizer, epoch, f1_score, path='checkpoints', filename='best_model.pt'):
    Path(path).mkdir(exist_ok=True)
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_score': f1_score
    }
    torch.save(checkpoint_data, Path(path) / filename)
    print(f"Saved checkpoint to {Path(path) / filename} with F1 score: {f1_score:.4f}")

def load_checkpoint(model, optimizer, path='checkpoints', filename='best_model.pt'):
    checkpoint_path = Path(path) / filename
    if checkpoint_path.exists():
        checkpoint_data = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        epoch = checkpoint_data['epoch']
        f1_score = checkpoint_data['f1_score']
        print(f"Loaded checkpoint from epoch {epoch} with F1 score: {f1_score:.4f}")
        return epoch, f1_score
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None


def train_epoch(model, loader, opt, loss_fn, epoch):
    model.train()
    tot_loss = 0
    metrics_collection = { # Renamed to avoid conflict
        'accuracy': torchmetrics.Accuracy('binary').to(DEVICE), 'precision': torchmetrics.Precision('binary').to(DEVICE),
        'recall': torchmetrics.Recall('binary').to(DEVICE), 'f1': torchmetrics.F1Score('binary').to(DEVICE),
        'roc_auc': torchmetrics.AUROC('binary').to(DEVICE), 'pr_auc': torchmetrics.AveragePrecision('binary').to(DEVICE),
        'mcc': torchmetrics.MatthewsCorrCoef('binary').to(DEVICE),
        'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)
    }
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=True)
    for g in pbar:
        g = g.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(g)
            loss = loss_fn(logits, g.y_node)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt); scaler.update()
        tot_loss += loss.item()
        binary_labels = (g.y_node > 0.5).long()
        for metric_obj in metrics_collection.values(): # Corrected variable name
            metric_obj.update(logits, binary_labels)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        wandb.log({"batch_loss": loss.item(), "learning_rate": opt.param_groups[0]['lr'], "epoch": epoch})

    log_metrics = {"train_loss": tot_loss / len(loader), "epoch": epoch, "learning_rate": opt.param_groups[0]['lr']}
    for name, metric_obj in metrics_collection.items(): # Corrected variable name
        log_metrics[f"train_{name}"] = metric_obj.compute().item()
        metric_obj.reset()
    conf_matrix = metrics_collection['confusion'].compute() # Compute once after iterating
    log_metrics.update({"train_tp": conf_matrix[1,1].item(), "train_fp": conf_matrix[0,1].item(),
                        "train_tn": conf_matrix[0,0].item(), "train_fn": conf_matrix[1,0].item()})
    metrics_collection['confusion'].reset() # Reset confusion matrix too
    wandb.log(log_metrics)
    print(f"Train Epoch {epoch}: Loss={log_metrics['train_loss']:.4f}, F1={log_metrics['train_f1']:.3f}, AUC={log_metrics['train_roc_auc']:.3f}")
    return log_metrics['train_loss']


@torch.no_grad()
def eval_epoch(model, loader, epoch, current_best_f1_for_checkpointing, optim): # Added for checkpointing
    model.eval()
    metrics_collection = { # Renamed
        'accuracy': torchmetrics.Accuracy('binary').to(DEVICE), 'precision': torchmetrics.Precision('binary').to(DEVICE),
        'recall': torchmetrics.Recall('binary').to(DEVICE), 'f1': torchmetrics.F1Score('binary').to(DEVICE),
        'roc_auc': torchmetrics.AUROC('binary').to(DEVICE), 'pr_auc': torchmetrics.AveragePrecision('binary').to(DEVICE),
        'mcc': torchmetrics.MatthewsCorrCoef('binary').to(DEVICE),
        'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)
    }
    pbar = tqdm(loader, desc=f'Eval Epoch {epoch}', leave=True)
    for g in pbar:
        g = g.to(DEVICE)
        with autocast(device_type="cuda", dtype=torch.float16): logits = model(g)
        binary_labels = (g.y_node > 0.5).long()
        for metric_obj in metrics_collection.values(): # Corrected variable name
            metric_obj.update(logits, binary_labels)
    
    log_metrics = {"epoch": epoch}
    for name, metric_obj in metrics_collection.items(): # Corrected variable name
        value = metric_obj.compute().item()
        log_metrics[f"val_{name}"] = value
        metric_obj.reset()
    conf_matrix = metrics_collection['confusion'].compute()
    log_metrics.update({"val_tp": conf_matrix[1,1].item(), "val_fp": conf_matrix[0,1].item(),
                        "val_tn": conf_matrix[0,0].item(), "val_fn": conf_matrix[1,0].item()})
    metrics_collection['confusion'].reset()
    wandb.log(log_metrics)
    print(f"Eval Epoch {epoch}: F1={log_metrics['val_f1']:.3f}, AUC={log_metrics['val_roc_auc']:.3f}, MCC={log_metrics['val_mcc']:.3f}")

    # Checkpoint saving logic moved here to ensure f1 is from current eval
    f1_for_checkpoint = log_metrics['val_f1']
    if f1_for_checkpoint > current_best_f1_for_checkpointing:
        # current_best_f1_for_checkpointing = f1_for_checkpoint # This will be updated in main loop
        save_checkpoint(model, optim, epoch, f1_for_checkpoint)
        print(f"New best model saved (during eval_epoch)! F1: {f1_for_checkpoint:.4f}")
        
    return log_metrics['val_roc_auc'], log_metrics['val_f1'], log_metrics['val_mcc']


@torch.no_grad()
def find_optimal_logit_threshold(model, loader, device, target_metric_name='f1', num_threshold_steps=200):
    model.eval()
    all_logits = []
    all_labels = []
    print(f"Finding optimal threshold for {target_metric_name}...")
    for g in tqdm(loader, desc="Collecting logits/labels for threshold search"):
        g = g.to(device)
        with autocast(device_type="cuda", dtype=torch.float16): # if using amp
            logits = model(g)
        all_logits.append(logits.cpu())
        all_labels.append(g.y_node.cpu().long()) # Ensure labels are long

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    min_logit, max_logit = all_logits.min().item(), all_logits.max().item()
    if min_logit == max_logit: # Handle case with no variance in logits
        print(f"Warning: All logits are the same ({min_logit}). Using 0.0 as threshold.")
        thresholds_to_test = np.array([0.0])
    else:
        thresholds_to_test = np.linspace(min_logit, max_logit, num_threshold_steps)

    best_metric_val = -float('inf')
    optimal_threshold = thresholds_to_test[0] if len(thresholds_to_test) > 0 else 0.0
    
    f1_computer = torchmetrics.F1Score('binary').to(device)
    mcc_computer = torchmetrics.MatthewsCorrCoef('binary').to(device)
    
    # Move all data to device once for faster metric computation if VRAM allows
    # Otherwise, compute in chunks or keep on CPU and move preds/labels per threshold
    all_logits_dev = all_logits.to(device)
    all_labels_dev = all_labels.to(device)

    for thresh in tqdm(thresholds_to_test, desc="Sweeping thresholds"):
        preds = (all_logits_dev >= thresh).long()
        
        current_metric_val = 0.0
        if target_metric_name == 'f1':
            current_metric_val = f1_computer(preds, all_labels_dev).item()
        elif target_metric_name == 'mcc':
            current_metric_val = mcc_computer(preds, all_labels_dev).item()
        else:
            raise ValueError(f"Unsupported target_metric_name: {target_metric_name}")

        if current_metric_val > best_metric_val:
            best_metric_val = current_metric_val
            optimal_threshold = thresh
    
    # Calculate all metrics at the optimal threshold
    optimal_preds = (all_logits_dev >= optimal_threshold).long()
    final_f1 = f1_computer(optimal_preds, all_labels_dev).item()
    final_mcc = mcc_computer(optimal_preds, all_labels_dev).item()
    
    f1_computer.reset()
    mcc_computer.reset()

    print(f"Optimal logit threshold: {optimal_threshold:.4f}")
    print(f"  Yields F1: {final_f1:.4f}")
    print(f"  Yields MCC: {final_mcc:.4f}")
    return optimal_threshold, final_f1, final_mcc


@torch.no_grad()
def evaluate_with_fixed_logit_threshold(model, loader, device, logit_threshold, id_string="final_eval"):
    model.eval()
    metrics_collection = {
        'accuracy': torchmetrics.Accuracy('binary').to(device),
        'precision': torchmetrics.Precision('binary').to(device),
        'recall': torchmetrics.Recall('binary').to(device),
        'f1': torchmetrics.F1Score('binary').to(device),
        'mcc': torchmetrics.MatthewsCorrCoef('binary').to(device),
        'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(device),
        # These metrics should use raw logits/probabilities, not binarized predictions
        'roc_auc': torchmetrics.AUROC('binary').to(device),
        'pr_auc': torchmetrics.AveragePrecision('binary').to(device),
    }
    
    pbar = tqdm(loader, desc=f'Evaluation ({id_string}) with threshold {logit_threshold:.3f}', leave=True)
    for g in pbar:
        g = g.to(device)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(g)
        
        binary_labels = (g.y_node > 0.5).long()
        # For threshold-dependent metrics
        binary_preds = (logits >= logit_threshold).long()

        metrics_collection['accuracy'].update(binary_preds, binary_labels)
        metrics_collection['precision'].update(binary_preds, binary_labels)
        metrics_collection['recall'].update(binary_preds, binary_labels)
        metrics_collection['f1'].update(binary_preds, binary_labels)
        metrics_collection['mcc'].update(binary_preds, binary_labels)
        metrics_collection['confusion'].update(binary_preds, binary_labels)
        
        # For threshold-agnostic metrics that operate on scores/logits
        metrics_collection['roc_auc'].update(logits, binary_labels)
        metrics_collection['pr_auc'].update(logits, binary_labels)

    log_metrics = {f"{id_string}_logit_threshold": logit_threshold}
    for name, metric_obj in metrics_collection.items():
        value = metric_obj.compute().item()
        log_metrics[f"{id_string}_{name}"] = value
        metric_obj.reset()
    
    conf_matrix_val = metrics_collection['confusion'].compute() # get value before reset
    metrics_collection['confusion'].reset() # ensure reset if not done above
    log_metrics.update({
        f"{id_string}_tp": conf_matrix_val[1,1].item(), f"{id_string}_fp": conf_matrix_val[0,1].item(),
        f"{id_string}_tn": conf_matrix_val[0,0].item(), f"{id_string}_fn": conf_matrix_val[1,0].item()
    })
    
    if wandb.run: wandb.log(log_metrics) # Log to wandb if active
    
    print(f"\n--- Evaluation Results ({id_string} with Logit Threshold: {logit_threshold:.4f}) ---")
    print(f"  F1: {log_metrics[f'{id_string}_f1']:.4f}")
    print(f"  AUROC: {log_metrics[f'{id_string}_roc_auc']:.4f}")
    print(f"  PR-AUC: {log_metrics[f'{id_string}_pr_auc']:.4f}")
    print(f"  MCC: {log_metrics[f'{id_string}_mcc']:.4f}")
    print(f"  Accuracy: {log_metrics[f'{id_string}_accuracy']:.4f}")
    print(f"  Precision: {log_metrics[f'{id_string}_precision']:.4f}")
    print(f"  Recall: {log_metrics[f'{id_string}_recall']:.4f}")
    print(f"  TP: {log_metrics[f'{id_string}_tp']}, FP: {log_metrics[f'{id_string}_fp']}, TN: {log_metrics[f'{id_string}_tn']}, FN: {log_metrics[f'{id_string}_fn']}")
    print("--- End of Evaluation ---")
    
    return log_metrics


# ---------------- 10. MAIN ---------------------------------
if __name__ == '__main__':
    if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    EDF_DIR  = '/media/dan/Data/data/baseline_patients/baseline_edfs' # Replace with your path
    CSV_PATH = 'contact_info.csv' # Replace with your path
    # Ensure these paths are correct or files exist, otherwise script will fail.
    # For demonstration, using dummy paths if real ones don't exist
    if not Path(EDF_DIR).exists() or not Path(CSV_PATH).exists():
        print(f"Warning: EDF_DIR ('{EDF_DIR}') or CSV_PATH ('{CSV_PATH}') not found. Using dummy data.")
        # Create dummy CSV
        Path("dummy_contact_info.csv").write_text("pid,electrode,X_recoded_dl2,Y_recoded_dl2,Z_recoded_dl2,soz\n1,CH1,0,0,0,1\n1,CH2,1,1,1,0")
        CSV_PATH = "dummy_contact_info.csv"
        # Create dummy EDF - this is harder, so we'll just limit PIDs to make it run
        TRAIN_P = [1] 
        TEST_P = [1]
        # The build_patient_graph would need a dummy EDF file too.
        # This part is complex to fully mock. Assuming files exist for a real run.
        # For a true dry run without data, you'd mock build_patient_graph.
        # Fallback to CPU if dummy data is used and no CUDA for MNE ops
        if not torch.cuda.is_available(): DEVICE = 'cpu' 
    else: # Original PIDs
        TRAIN_P  = [6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 26, 27, 28, 30, 31, 33, 34, 35, 36, 39, 40, 43, 47, 51, 55, 60, 62, 64, 66, 69, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 86, 87, 89, 90, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102]
        TEST_P   = [105, 106, 108, 109, 111, 112, 113]


    wandb.init(
        project="ez-gnn",
        config={
            "learning_rate": 3e-4, "weight_decay": 3e-4, "epochs": 200, # Max epochs
            "batch_size": 1, "model_name": "EZGNN_v2_minRecipe", # Updated name
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR_Tmax50",
            "loss": "BCEWithLogitsLoss_pos_weight7", "dropout": 0.25,
            "gnn_layers": LAYERS, "hidden_dim": HID_DIM,
            "early_stopping_patience_evals": 6 # Number of evaluations, not epochs
        }
    )
    
    torch.set_grad_enabled(True)
    
    train_ds = EZDataset(TRAIN_P, EDF_DIR, CSV_PATH)
    # Assuming TEST_P is your validation set for now
    val_ds  = EZDataset(TEST_P,  EDF_DIR, CSV_PATH) 
    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=wandb.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = EZGNN().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50) # T_max from minimal recipe
    pos_weight_val = torch.tensor([7.0], device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    print(f"Training Config: LR={wandb.config.learning_rate}, WD={wandb.config.weight_decay}, Epochs={wandb.config.epochs}")
    
    best_f1_for_checkpointing = 0.0 # Tracks overall best F1 for saving model
    best_val_f1_for_early_stopping = 0.0
    early_stopping_patience_counter = 0
    
    # Corrected variable for f1 from eval_epoch
    f1_from_eval = 0.0

    for epoch in range(1, wandb.config.epochs + 1):
        print(f"\nEpoch {epoch}/{wandb.config.epochs}")
        train_loss = train_epoch(model, train_loader, optim, loss_fn, epoch)
        scheduler.step()

        if epoch == 1 or epoch % 3 == 0:
            # Pass current best_f1 for checkpointing and optimizer to eval_epoch
            val_auroc, f1_from_eval, val_mcc = eval_epoch(model, val_loader, epoch, best_f1_for_checkpointing, optim)
            
            print(f"Epoch {epoch} Val Results: Train Loss={train_loss:.4f}, Val AUROC={val_auroc:.3f}, Val F1={f1_from_eval:.3f} (default thresh)")
            wandb.log({"epoch_val_f1_default_thresh": f1_from_eval, 
                       "epoch_val_auroc": val_auroc, 
                       "epoch_val_mcc_default_thresh": val_mcc,
                       "epoch_lr": scheduler.get_last_lr()[0],
                       "epoch": epoch})

            if f1_from_eval > best_f1_for_checkpointing: # This is for saving the model
                 best_f1_for_checkpointing = f1_from_eval
                 # save_checkpoint is now called inside eval_epoch if f1 is better

            if f1_from_eval > best_val_f1_for_early_stopping:
                best_val_f1_for_early_stopping = f1_from_eval
                early_stopping_patience_counter = 0
            else:
                early_stopping_patience_counter += 1
            
            print(f"Early stopping counter: {early_stopping_patience_counter}/{wandb.config.early_stopping_patience_evals}")
            if early_stopping_patience_counter >= wandb.config.early_stopping_patience_evals:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print("\nTraining finished or early stopping triggered.")
    
    # --- Post-training: Load best model and find optimal threshold ---
    print("\n--- Starting Post-Training Evaluation ---")
    best_model_path = Path('checkpoints') / 'best_model.pt'
    if best_model_path.exists():
        print("Loading best model for final evaluation...")
        # Re-initialize model and optimizer to load state
        # (optimizer state not strictly needed for eval but good practice for load_checkpoint)
        final_model = EZGNN().to(DEVICE)
        # Dummy optimizer for loading, not used for further training
        # final_optim = torch.optim.AdamW(final_model.parameters(), lr=wandb.config.learning_rate) 
        # load_checkpoint(final_model, final_optim, path='checkpoints', filename='best_model.pt')
        # Simpler: just load model state_dict if optimizer state isn't crucial for eval
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch','N/A')} with F1 (default thresh): {checkpoint.get('f1_score','N/A'):.4f}")

        optimal_logit_thresh, f1_at_optimal, mcc_at_optimal = find_optimal_logit_threshold(
            final_model, val_loader, DEVICE, target_metric_name='f1' # Can change to 'mcc'
        )
        wandb.log({
            "optimal_validation_logit_threshold_f1": optimal_logit_thresh,
            "validation_f1_at_optimal_threshold": f1_at_optimal,
            "validation_mcc_at_optimal_f1_threshold": mcc_at_optimal,
        })
        
        print(f"\nEvaluating on Validation Set with Optimal Logit Threshold: {optimal_logit_thresh:.4f}")
        evaluate_with_fixed_logit_threshold(
            final_model, val_loader, DEVICE, optimal_logit_thresh, id_string="val_optimal_thresh"
        )
        
        # If you have a separate true test set (e.g., test_ds, test_loader_final):
        # print(f"\nEvaluating on Test Set with Optimal Logit Threshold: {optimal_logit_thresh:.4f}")
        # evaluate_with_fixed_logit_threshold(
        #     final_model, test_loader_final, DEVICE, optimal_logit_thresh, id_string="test_optimal_thresh"
        # )
    else:
        print("No best model checkpoint found to evaluate.")

    wandb.finish()
    print("Run finished.")