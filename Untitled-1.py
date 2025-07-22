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
HID_DIM      = 16
LAYERS       = 4
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
            z_win = checkpoint(self.temporal, win.reshape(-1,1,L).contiguous())
            z_edge = self.tpool(z_win.view(B, W, Z_DIM))
            edge_attr.append(z_edge)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.cat(edge_attr, 0)                        # (E,Z_DIM)

    # ---- forward -------------------------------------------
    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(self._compute_edge_attr(data.signals, data.edge_index))
        for gnn in self.gnn_layers:
            x = gnn(x, data.edge_index, edge_attr)
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
# (unchanged – omitted here for brevity)
# …

# ---------------- 8.  DATASET -------------------------------
# (unchanged – omitted here for brevity)
# …

# ---------------- 9.  TRAIN / EVAL LOOPS --------------------

def train_epoch(model, loader, opt, loss_fn, epoch):
    model.train(); tot_loss = 0
    metrics = {'accuracy': torchmetrics.Accuracy('binary').to(DEVICE),
               'f1':        torchmetrics.F1Score('binary').to(DEVICE),
               'roc_auc':   torchmetrics.AUROC('binary').to(DEVICE),
               'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)}
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    start = time.time()
    for g in pbar:
        g = g.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(g)
            loss   = loss_fn(logits, g.y_node)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                               # <‑‑ patch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(opt); scaler.update()
        tot_loss += loss.item()
        for m in metrics.values():
            m.update(logits, (g.y_node>0.5).long())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    res = {k: m.compute().item() for k,m in metrics.items() if k!='confusion'}
    cm  = metrics['confusion'].compute();  res.update({'tp': cm[1,1].item(),'fp':cm[0,1].item(),'tn':cm[0,0].item(),'fn':cm[1,0].item()})
    wandb.log({'train_loss': tot_loss/len(loader), **res, 'epoch': epoch})
    for m in metrics.values(): m.reset()
    print(f"\nEpoch {epoch} Summary: time {time.time()-start:.1f}s | loss {tot_loss/len(loader):.4f} | F1 {res['f1']:.3f} | AUC {res['roc_auc']:.3f}")
    return tot_loss/len(loader)

@torch.no_grad()
def eval_epoch(model, loader, epoch):
    model.eval(); metrics = {'accuracy': torchmetrics.Accuracy('binary').to(DEVICE),
                             'f1':        torchmetrics.F1Score('binary').to(DEVICE),
                             'roc_auc':   torchmetrics.AUROC('binary').to(DEVICE),
                             'confusion': torchmetrics.ConfusionMatrix('binary', num_classes=2).to(DEVICE)}
    for g in tqdm(loader, desc=f'Validation {epoch}'):
        g = g.to(DEVICE)
        logits = model(g)
        for m in metrics.values():
            m.update(logits, (g.y_node>0.5).long())
    res = {k: m.compute().item() for k,m in metrics.items() if k!='confusion'}
    cm  = metrics['confusion'].compute();  res.update({'tp': cm[1,1].item(),'fp':cm[0,1].item(),'tn':cm[0,0].item(),'fn':cm[1,0].item()})
    wandb.log({f'val_{k}':v for k,v in res.items()} | {'epoch':epoch})
    for m in metrics.values(): m.reset()
    print(f"Validation: Acc {res['accuracy']:.3f} | F1 {res['f1']:.3f} | AUC {res['roc_auc']:.3f}")
    return res['roc_auc'], res['f1']

# ---------------- 10.  MAIN --------------------------------
if __name__ == '__main__':
    EDF_DIR  = '/media/dan/Data/data/baseline_patients/baseline_edfs'
    CSV_PATH = 'contact_info.csv'
    TRAIN_P  = [...]  # unchanged
    TEST_P   = [...]

    wandb.init(project='ez-gnn', config=dict(lr=3e-4, epochs=500, batch_size=1))

    train_loader = DataLoader(EZDataset(TRAIN_P, EDF_DIR, CSV_PATH), batch_size=1, shuffle=True)
    test_loader  = DataLoader(EZDataset(TEST_P,  EDF_DIR, CSV_PATH), batch_size=1)

    model = EZGNN().to(DEVICE)
    pos_weight = torch.tensor([7.0], device=DEVICE)  # ~~ TN/TP ratio
    loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn  = FocalBCELoss(alpha=0.25, gamma=2.0)   # <‑‑ keep for experiments

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_f1 = 0.0
    for epoch in range(1, 501):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, epoch)
        if epoch % 5 == 0 or epoch == 1:
            auc, f1 = eval_epoch(model, test_loader, epoch)
            if f1 > best_f1:
                best_f1 = f1
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optim_state_dict':optim.state_dict(),'f1':f1}, 'checkpoints/best_model.pt')
                print(f"\nNew best model saved @ epoch {epoch} | F1={f1:.4f}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    wandb.finish()
