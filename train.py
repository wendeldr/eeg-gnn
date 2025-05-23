# -----------------------------------------------------------
# 4. TEMPORAL ENCODER  (shared across every edge window)
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

# ---- 4a. Dilated-TCN block --------------------------------
class TemporalEncoder(nn.Module):
    """
    Input:  (B, 1, 2000)   # 2-s window @ 1 kHz
    Output: (B, 64)        # z_w  (window embedding)
    """
    def __init__(self, hid=32, out_dim=64, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  hid,  k, dilation=1, padding=(k-1)//2)
        self.conv2 = nn.Conv1d(hid, hid, k, dilation=2, padding=((k-1)//2)*2)
        self.conv3 = nn.Conv1d(hid, hid, k, dilation=4, padding=((k-1)//2)*4)
        self.conv4 = nn.Conv1d(hid, out_dim, k, dilation=8, padding=((k-1)//2)*8)
        self.act   = nn.ReLU()

    def forward(self, x):               # x: (B, 1, 2000)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))     # (B, 64, 2000)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # global-max
        return x                        # (B, 64)


# ---- 4b. Window-attention pooling -------------------------
class AttentionPool(nn.Module):
    """
    Pools a sequence of window embeddings to one edge vector.
    Input:  (N_w, 64)
    Output: (64,)
    """
    def __init__(self, dim=64):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))

    def forward(self, z):               # z: (N_w, 64)
        scores = torch.matmul(z, self.q)          # (N_w,)
        α = F.softmax(scores, dim=0).unsqueeze(1) # (N_w,1)
        return torch.sum(α * z, dim=0)            # (64,)


# -----------------------------------------------------------
# 5. EDGE-CONDITIONED MESSAGE-PASSING LAYER
# -----------------------------------------------------------
class EdgeMPNN(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='add')  # Σ-aggregation
        self.lin_src   = nn.Linear(in_channels,  out_channels)
        self.lin_dst   = nn.Linear(in_channels,  out_channels)
        self.edge_mlp  = nn.Sequential(
            nn.Linear(2*out_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # x: (N,H)   edge_attr: (E,Fe)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: dst, x_j: src
        msg_input = torch.cat([self.lin_src(x_j), self.lin_dst(x_i), edge_attr], dim=-1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.act(x + aggr_out)   # residual


# -----------------------------------------------------------
# 6. FULL MODEL
# -----------------------------------------------------------
class EZGNN(nn.Module):
    def __init__(self, node_in=40, edge_in=68, h=64, layers=4,
                 ilae_classes=5, use_ilae=True):
        super().__init__()
        self.use_ilae = use_ilae
        # --- node/edge input projections
        self.lin_node = nn.Linear(node_in, h)
        self.lin_edge = nn.Linear(edge_in, h)
        # --- MPNN stack
        self.blocks = nn.ModuleList([
            EdgeMPNN(h, h, h) for _ in range(layers)
        ])
        # --- heads
        self.node_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        if use_ilae:
            self.global_head = nn.Sequential(
                nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, ilae_classes)
            )

    def forward(self, data):
        # data: torch_geometric.data.Data
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x  = self.lin_node(x)
        ea = self.lin_edge(ea)
        for l, gnn in enumerate(self.blocks):
            x = gnn(x, ei, ea)
        # --- node logits
        node_logit = torch.sigmoid(self.node_head(x)).view(-1)
        if not self.use_ilae:
            return node_logit, None
        # --- patient logits
        g = torch_geometric.nn.global_mean_pool(x, batch)   # (P,H)
        pat_logit = self.global_head(g)                     # (P,5)
        return node_logit, pat_logit


# -----------------------------------------------------------
# 7. FOCAL-BCE + (optional) ILAE LOSS
# -----------------------------------------------------------
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, target):
        eps = 1e-6
        pt   = torch.where(target==1, logits, 1-logits).clamp(eps, 1-eps)
        w    = self.alpha * (1-pt).pow(self.gamma)
        loss = -w * torch.where(target==1, torch.log(logits+eps),
                                            torch.log(1-logits+eps))
        return loss.mean() if self.reduction=='mean' else loss


def compute_loss(node_logits, node_labels,
                 pat_logits, pat_labels,
                 λ_node=1.0, λ_pat=0.3, alpha=0.25, gamma=2.0):
    criterion_node = FocalBCELoss(alpha, gamma)
    node_loss = criterion_node(node_logits, node_labels.float())
    if pat_logits is None:
        return node_loss
    ce = nn.CrossEntropyLoss()
    pat_loss = ce(pat_logits, pat_labels.long())
    return λ_node*node_loss + λ_pat*pat_loss
