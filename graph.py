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
HID_DIM      = 8
LAYERS       = 3
BASE_EDGE_CHUNK = 32
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {DEVICE}")

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
        self.dropout_layers = torch.nn.ModuleList([torch.nn.Dropout(0.25) for _ in range(LAYERS)])
        self.out_head   = torch.nn.Sequential(
            torch.nn.LayerNorm(HID_DIM),
            torch.nn.Linear(HID_DIM, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    # ---- helper: edge embedding in chunks ------------------
    def _compute_edge_attr(self, signals, edge_index):
        E = edge_index.size(1)
        N = signals.size(0)
        edge_chunk = int(max(16, BASE_EDGE_CHUNK * (100/max(N,1)))) if N > 150 else BASE_EDGE_CHUNK
        edge_attr = []
        for s in range(0, E, edge_chunk):
            idx  = edge_index[:, s:s+edge_chunk]
            diff = signals[idx[0]] - signals[idx[1]]          # (B,T)
            win  = diff.unfold(-1, WIN_LEN, STRIDE)            # (B,W,L)
            B, W, L = win.shape
            # Ensure inputs have gradients and use explicit use_reentrant parameter
            win_reshaped = win.reshape(-1,1,L).contiguous().requires_grad_(True)
            z_win = checkpoint(self.temporal, win_reshaped, use_reentrant=False)
            z_edge = self.tpool(z_win.view(B, W, Z_DIM))
            edge_attr.append(z_edge)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(edge_attr, 0)                        # (E,Z_DIM)

    # ---- forward -------------------------------------------
    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(self._compute_edge_attr(data.signals, data.edge_index))
        for i, gnn in enumerate(self.gnn_layers):
            x = gnn(x, data.edge_index, edge_attr)
            x = self.dropout_layers[i](x)
        logits = self.out_head(x).view(-1)   # <‑‑ raw logits, no sigmoid
        return logits

# ---------------- 6.  LOSSES --------------------------------
class FocalBCELoss(torch.nn.Module):
    """Kept for reference; not used in current training run"""
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

    # ---- load EDF + notch all 60 Hz harmonics --------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(os.path.join(edf_dir, f"{pid:03}_Baseline.EDF"),
                                  preload=True, verbose=False)
    assert int(raw.info['sfreq']) == FS_HZ
    sf = raw.info['sfreq']

    # ---- channel matching ----------------------------------
    pat = contact_df[contact_df.pid == pid].copy()
    raw_ch_no_prefix = [ch.replace('EEG ', '').strip() for ch in raw.ch_names if 'EEG ' in ch]
    pat['eeg_match'] = pat.electrode.str.strip()
    valid = pat[pat.eeg_match.isin(raw_ch_no_prefix)]
    picks_map = {row.eeg_match: f"EEG {row.electrode.strip()}" for _, row in valid.iterrows()}
    ordered_picks = [picks_map[n] for n in valid.eeg_match]

    # notch comb
    raw.notch_filter(np.arange(60, int(sf//2), 60), picks=ordered_picks,
                     filter_length='auto', phase='zero',
                     fir_window='hamming', fir_design='firwin', verbose=False)
    data = raw.get_data(picks=ordered_picks).astype(np.float32)     # (N,T)

    # cut data due to memory constraints
    data = data[:, :int(sf*10)] * 1e-6 # scale to uV

    # ---- node features  (xyz quantised → z-score) ----------
    xyz = valid[['X_recoded_dl2','Y_recoded_dl2','Z_recoded_dl2']].to_numpy(float)
    # xyz = (xyz - xyz.mean(0)) / xyz.std(0) # pre normalized not needed
    x_node = torch.from_numpy(xyz).float()

    # ---- edges (directed complete) -------------------------
    N = x_node.size(0)
    pairs = list(combinations(range(N), 2))
    src, dst = zip(*pairs)
    edge_idx = torch.tensor(np.vstack([src+dst, dst+src]), dtype=torch.long)

    # ---- labels -------------------------------------------
    y_node = torch.tensor(valid.soz.values, dtype=torch.float32)

    # ---- signals tensor -----------------------------------
    signals = torch.from_numpy(data)                # (N,T)

    # Add NaN checks for data
    # if torch.isnan(x_node).any(): print(f"NaNs found in x_node for pid {pid}")
    # if torch.isnan(signals).any(): print(f"NaNs found in signals for pid {pid}")
    # if torch.isnan(y_node).any(): print(f"NaNs found in y_node for pid {pid}")

    return Data(x=x_node,
                edge_index=edge_idx,
                y_node=y_node,
                signals=signals,
                pid=torch.tensor([pid]))


# ---------------- 8.  DATASET -------------------------------
class EZDataset(torch.utils.data.Dataset):
    def __init__(self, pids, edf_dir, contact_csv):
        self.pids, self.edf_dir, self.contact_csv = pids, edf_dir, contact_csv
    def __len__(self):  return len(self.pids)
    def __getitem__(self, idx):
        return build_patient_graph(self.pids[idx], self.edf_dir, self.contact_csv)


# ---------------- 9.  TRAIN / EVAL LOOPS --------------------
def save_checkpoint(model, optimizer, epoch, f1_score, path='checkpoints'):
    """Save model checkpoint if it's the best so far"""
    Path(path).mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_score': f1_score
    }
    torch.save(checkpoint, f'{path}/best_model.pt')
    print(f"Saved checkpoint with F1 score: {f1_score:.4f}")

def train_epoch(model, loader, opt, loss_fn, epoch):
    model.train()
    tot_loss = 0
    
    # Initialize metrics
    metrics = {
        'accuracy': torchmetrics.Accuracy('binary').to(DEVICE),
        'precision': torchmetrics.Precision('binary').to(DEVICE),
        'recall': torchmetrics.Recall('binary').to(DEVICE),
        'f1': torchmetrics.F1Score('binary').to(DEVICE),
        'roc_auc': torchmetrics.AUROC('binary').to(DEVICE),
        'pr_auc': torchmetrics.AveragePrecision('binary').to(DEVICE),
        'mse': torchmetrics.MeanSquaredError().to(DEVICE),
        'mae': torchmetrics.MeanAbsoluteError().to(DEVICE),
        'r2': torchmetrics.R2Score().to(DEVICE),
        'mcc': torchmetrics.MatthewsCorrCoef('binary').to(DEVICE),
        'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)
    }
    
    # Create progress bar
    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=True)
    start_time = time.time()
    
    for batch_idx, g in enumerate(pbar):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        g = g.to(DEVICE)
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(g)
            loss = loss_fn(logits, g.y_node)

        scaler.scale(loss).backward()
        scaler.unscale_(opt) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt)
        scaler.update()
        tot_loss += loss.item()
        
        # Convert labels to binary format for metrics
        binary_labels = (g.y_node > 0.5).long()
        
        # Update all metrics
        for metric in metrics.values():
            metric.update(logits, binary_labels)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
        
        # Log batch metrics
        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": opt.param_groups[0]['lr'],
            "epoch": epoch
        })
    
    # Compute and log all metrics
    metric_results = {
        "train_loss": tot_loss / len(loader),
        "train_accuracy": metrics['accuracy'].compute().item(),
        "train_precision": metrics['precision'].compute().item(),
        "train_recall": metrics['recall'].compute().item(),
        "train_f1": metrics['f1'].compute().item(),
        "train_roc_auc": metrics['roc_auc'].compute().item(),
        "train_pr_auc": metrics['pr_auc'].compute().item(),
        "train_mse": metrics['mse'].compute().item(),
        "train_mae": metrics['mae'].compute().item(),
        "train_r2": metrics['r2'].compute().item(),
        "train_mcc": metrics['mcc'].compute().item(),
        "epoch": epoch,
        "learning_rate": opt.param_groups[0]['lr']
    }
    
    # Get confusion matrix and extract TP, FP, TN, FN
    conf_matrix = metrics['confusion'].compute()
    metric_results.update({
        "train_tp": conf_matrix[1, 1].item(),
        "train_fp": conf_matrix[0, 1].item(),
        "train_tn": conf_matrix[0, 0].item(),
        "train_fn": conf_matrix[1, 0].item()
    })
    
    # Log epoch metrics
    wandb.log(metric_results)
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    # Print epoch summary
    elapsed_time = time.time() - start_time
    print(f"\nEpoch {epoch} Summary:")
    print(f"Time: {elapsed_time:.2f}s | Loss: {tot_loss/len(loader):.4f}")
    print(f"Metrics: Acc={metric_results['train_accuracy']:.3f} | F1={metric_results['train_f1']:.3f} | AUC={metric_results['train_roc_auc']:.3f}")
    
    return tot_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, epoch, prefix=""):
    model.eval()
    
    # Initialize metrics
    metrics = {
        'accuracy': torchmetrics.Accuracy('binary').to(DEVICE),
        'precision': torchmetrics.Precision('binary').to(DEVICE),
        'recall': torchmetrics.Recall('binary').to(DEVICE),
        'f1': torchmetrics.F1Score('binary').to(DEVICE),
        'roc_auc': torchmetrics.AUROC('binary').to(DEVICE),
        'pr_auc': torchmetrics.AveragePrecision('binary').to(DEVICE),
        'mse': torchmetrics.MeanSquaredError().to(DEVICE),
        'mae': torchmetrics.MeanAbsoluteError().to(DEVICE),
        'r2': torchmetrics.R2Score().to(DEVICE),
        'mcc': torchmetrics.MatthewsCorrCoef('binary').to(DEVICE),
        'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)
    }
    
    # Create progress bar
    pbar = tqdm(loader, desc=f'{prefix} Validation {epoch}', leave=True)
    start_time = time.time()
    
    for g in pbar:
        g = g.to(DEVICE)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(g)
        
        # Convert labels to binary format for metrics
        binary_labels = (g.y_node > 0.5).long()
        
        # Update all metrics
        for metric in metrics.values():
            metric.update(logits, binary_labels)
    
    # Compute and log all metrics
    metric_results = {
        f"{prefix}_accuracy": metrics['accuracy'].compute().item(),
        f"{prefix}_precision": metrics['precision'].compute().item(),
        f"{prefix}_recall": metrics['recall'].compute().item(),
        f"{prefix}_f1": metrics['f1'].compute().item(),
        f"{prefix}_roc_auc": metrics['roc_auc'].compute().item(),
        f"{prefix}_pr_auc": metrics['pr_auc'].compute().item(),
        f"{prefix}_mse": metrics['mse'].compute().item(),
        f"{prefix}_mae": metrics['mae'].compute().item(),
        f"{prefix}_r2": metrics['r2'].compute().item(),
        f"{prefix}_mcc": metrics['mcc'].compute().item(),
        "epoch": epoch
    }
    
    # Get confusion matrix and extract TP, FP, TN, FN
    conf_matrix = metrics['confusion'].compute()
    metric_results.update({
        f"{prefix}_tp": conf_matrix[1, 1].item(),
        f"{prefix}_fp": conf_matrix[0, 1].item(),
        f"{prefix}_tn": conf_matrix[0, 0].item(),
        f"{prefix}_fn": conf_matrix[1, 0].item()
    })
    
    # Log validation metrics
    wandb.log(metric_results)
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    # Print validation summary
    elapsed_time = time.time() - start_time
    print(f"\n{prefix} Validation Summary:")
    print(f"Time: {elapsed_time:.2f}s")
    print(f"Metrics: Acc={metric_results[f'{prefix}_accuracy']:.3f} | F1={metric_results[f'{prefix}_f1']:.3f} | AUC={metric_results[f'{prefix}_roc_auc']:.3f}")
    
    return metric_results[f"{prefix}_roc_auc"], metric_results[f"{prefix}_f1"]

# ---------------- 10. MAIN ---------------------------------
if __name__ == '__main__':
    # Set random seeds for reproducibility
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    EDF_DIR  = '/media/dan/Data/data/baseline_patients/baseline_edfs'
    CSV_PATH = 'contact_info.csv'
    
    # Read unique patient IDs from CSV
    contact_df = pd.read_csv(CSV_PATH)
    ALL_PATIENTS = sorted(contact_df['pid'].unique().tolist())
    
    # Create train/val/test split (60/20/20)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(ALL_PATIENTS)
    n = len(ALL_PATIENTS)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    TRAIN_P = ALL_PATIENTS[:train_size]
    VAL_P = ALL_PATIENTS[train_size:train_size + val_size]
    TEST_P = ALL_PATIENTS[train_size + val_size:]

    # Initialize wandb
    wandb.init(
        project="ez-gnn",
        config={
            "learning_rate": 3e-4,
            "weight_decay": 3e-4,
            "epochs": 200,
            "batch_size": 1,
            "model": "EZGNN_v2_minimal_recipe",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR_Tmax50",
            "loss": "BCEWithLogitsLoss_pos_weight7",
            "dropout": 0.25,
            "gnn_layers": 3,
            "hidden_dim": 8,
            "early_stopping_patience": 6,
            "random_seed": RANDOM_SEED,
            "train_size": len(TRAIN_P),
            "val_size": len(VAL_P),
            "test_size": len(TEST_P)
        }
    )
    
    # Enable gradient checkpointing
    torch.set_grad_enabled(True)
    
    train_ds = EZDataset(TRAIN_P, EDF_DIR, CSV_PATH)
    val_ds = EZDataset(VAL_P, EDF_DIR, CSV_PATH)
    test_ds = EZDataset(TEST_P, EDF_DIR, CSV_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = EZGNN().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)
    pos_weight = torch.tensor([10.0], device=DEVICE)  # ~~ TN/TP ratio
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining Configuration:")
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Train patients: {len(TRAIN_P)}")
    print(f"Validation patients: {len(VAL_P)}")
    print(f"Test patients: {len(TEST_P)}")
    print(f"Learning rate: {optim.param_groups[0]['lr']:.2e}")
    print(f"Batch size: 1")
    print(f"Model: EZGNN")
    print(f"Optimizer: AdamW")
    print(f"Weight decay: {optim.param_groups[0]['weight_decay']:.2e}")
    print(f"Scheduler: CosineAnnealingLR (T_max=50)")
    print(f"Loss: BCEWithLogitsLoss (pos_weight=10.0)")
    print("-" * 50)
    
    best_f1 = 0
    best_val_f1_for_early_stopping = 0.0
    early_stopping_patience = 6
    early_stopping_counter = 0
    
    for epoch in range(1, 200):
        print(f"\nEpoch {epoch}/200")
        train_loss = train_epoch(model, train_loader, optim, loss_fn, epoch)
        
        # Step the scheduler after training
        scheduler.step()

        if epoch == 1 or epoch % 3 == 0:
            # Validation
            val_auroc, val_f1 = eval_epoch(model, val_loader, epoch, prefix="val")
            
            # Test (only log metrics, don't use for early stopping)
            test_auroc, test_f1 = eval_epoch(model, test_loader, epoch, prefix="test")
            
            print(f"\nEpoch {epoch} Results:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val AUROC: {val_auroc:.3f} | Val F1: {val_f1:.3f}")
            print(f"Test AUROC: {test_auroc:.3f} | Test F1: {test_f1:.3f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping logic based on validation F1
            if val_f1 > best_val_f1_for_early_stopping:
                best_val_f1_for_early_stopping = val_f1
                early_stopping_counter = 0
                # Save best model based on validation F1
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    save_checkpoint(model, optim, epoch, best_f1)
                    print(f"\nNew best model saved! F1: {best_f1:.4f}")
            else:
                early_stopping_counter += 1
            
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} due to no improvement in validation F1 for {early_stopping_patience} evaluations.")
                break
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation on test set
    print("\nFinal Test Set Evaluation:")
    test_auroc, test_f1 = eval_epoch(model, test_loader, epoch, prefix="final_test")
    print(f"Final Test AUROC: {test_auroc:.3f} | Final Test F1: {test_f1:.3f}")
    
    wandb.finish()
