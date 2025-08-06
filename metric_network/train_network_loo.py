import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import fbeta_score, roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from termcolor import colored
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool  # we won’t pool, but handy later
from torch_geometric.nn import GATConv

import copy


def build_patient_graphs(csv_path: str,
                         skip_cols: list[str]) -> tuple[list[Data], list[str]]:

    # df = pd.read_csv(csv_path, low_memory=False)
    df = pd.read_feather('/media/dan/Data/git/epd_network/renamed_mean_data.feather') 
    
    # we add additional columns later so get them now
    # targets = [c for c in df.columns if c not in skip_cols]

    # these were from the feature selection notebook
    # targets = ["PLI_150-250","PLV_150-250","Phase_0-NYQ","Phase_1-4","Phase_1-70","Phase_13-30","Phase_30-70","Phase_70-150","bary_euclidean_max","bary_euclidean_mean","cov_sq-GraphicalLasso","dsPLI_1-250","dsPLI_1-4","dsPLI_150-250","dsPLI_30-70","dsPLI_70-250","dswPLI_0-NYQ","dswPLI_13-30","dswPLI_4-8","dswPLI_70-150","dswPLI_8-13","iCoh_0-NYQ","iCoh_13-30","iCoh_4-8","iCoh_70-150","je_gaussian","pec_orth_log","pec_orth_log_abs","prec_sq-EmpiricalCovariance","prec_sq-GraphicalLasso","prec_sq-ShrunkCovariance","xcorr_mean","xcorr_sq-mean"]

    # these are manually reviewed and "simplified" features
    targets = ['iCoh_1-250', 'pdist_cosine', 'bary_euclidean_mean', 'Phase_1-250', 'pec_orth', 'pec_orth_log_abs']

    # remove any patients with only one value of soz_sum
    df = df[df.groupby('pid')['soz_sum'].transform('nunique') > 1]

    inf_mask = (df[targets] > 1e200) | (df[targets] < -1e200)
    df.loc[:, targets] = df[targets].replace([np.inf, -np.inf], np.nan)
    df.loc[:, targets] = df[targets].mask(inf_mask, np.nan)
    df[targets] = df[targets].fillna(0)

    # fill miccai_label_a and miccai_label_b with 'unknown' if nan
    df['miccai_label_a'] = df['miccai_label_a'].fillna('unknown')
    df['miccai_label_b'] = df['miccai_label_b'].fillna('unknown')

    # fill etiology with 'unknown' if nan
    df['etiology'] = df['etiology'].fillna('Unassigned')

    # get all miccai_label_a and miccai_label_b values
    unique_miccai = list(sorted(np.unique(df[['miccai_label_a', 'miccai_label_b']].values.flatten())))
    miccai_to_idx = {l: i for i, l in enumerate(unique_miccai)}
    df['miccai_label_a_idx'] = df['miccai_label_a'].map(miccai_to_idx)
    df['miccai_label_b_idx'] = df['miccai_label_b'].map(miccai_to_idx)

    # etiology to idx
    unique_etiologies = list(sorted(np.unique(df['etiology'].values.flatten())))
    etiology_to_idx = {l: i for i, l in enumerate(unique_etiologies)}
    df['etiology_idx'] = df['etiology'].map(etiology_to_idx)

    # ── z‑score every feature inside each patient ──────────────────────────────
    df_z = df.copy()
    for col in targets:
        df_z[col] = (df.groupby("pid")[col]
                       .transform(lambda x: (x - x.mean()) / x.std(ddof=0)
                                  if x.std(ddof=0) != 0 else 0.0))

    # replace nan with 0
    df_z[targets] = df_z[targets].fillna(0)

    data_list = []
    for pid, grp in tqdm(df_z.groupby("pid"),desc="Building patient graphs"):
        n_nodes = int(max(grp.electrode_a.max(), grp.electrode_b.max()))

        # edge matrix
        edge_mat = np.zeros((n_nodes,n_nodes,len(targets)))
        a_idx = grp['electrode_a'].values.astype(int) - 1
        b_idx = grp['electrode_b'].values.astype(int) - 1
        target_vals = grp[targets].values  # shape (num_edges, num_targets)

        for (src, dst) in [(a_idx, b_idx), (b_idx, a_idx)]:
            edge_mat[src[:, None], dst[:, None], np.arange(len(targets))] = target_vals

        # Create a mask that is True on the diagonal
        mask2d = np.eye(edge_mat.shape[0], dtype=bool)
        mask3d = np.broadcast_to(mask2d[:, :, None], edge_mat.shape) 

        # Wrap A in a masked array, masking out the diagonal
        M = np.ma.masked_array(edge_mat, mask=mask3d)
        
        # Compute row‐wise mean and std, ignoring masked elements
        row_means = M.mean(axis=0).data
        row_stds  = M.std(axis=0).data

        edges, e_attr = [], []

        y = torch.full((n_nodes,), -1, dtype=torch.long)

        z = row_means.shape[1]
        node_lbls = torch.full((n_nodes,z*2 + 3), -1, dtype=torch.long)
        node_lbls[:,0:z] = torch.tensor(row_means)
        node_lbls[:,z:z*2] = torch.tensor(row_stds)

        # node labels

        # Collect edges and their target (154‑D) attributes
        for _, row in grp.iterrows():
            i, j = int(row.electrode_a) - 1, int(row.electrode_b) - 1
            feat = row[targets].to_numpy(dtype=np.float32)

            # apperently one direction is incorrect since it never lets the model learn the other direction
            # we need to add both directions
            # edges.append([i, j])
            # e_attr.append(feat)
            edges += [[i, j], [j, i]]
            e_attr += [feat, feat]

            # Node labels (only need to be set once)
            if y[i] == -1:
                y[i] = int(row.soz_a)
                node_lbls[i,-3:] = torch.tensor([row.miccai_label_a_idx, row.age_days, row.etiology_idx])
                
            if y[j] == -1:
                y[j] = int(row.soz_b)
                node_lbls[j,-3:] = torch.tensor([row.miccai_label_b_idx, row.age_days, row.etiology_idx])
        
        edges = np.array(edges)
        e_attr = np.array(e_attr)
        edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(e_attr,dtype=torch.float32).contiguous()

        # Use node_lbls as node features (shape [n_nodes, x])
        x = node_lbls.float()

        ilae_label = torch.tensor([grp.ilae.values[0]], dtype=torch.float32)

        data_list.append(Data(x=x,
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              y=y,
                              pid=torch.tensor([pid]),
                              ilae=ilae_label))   # keep pid if you need it

    return data_list, targets



class NodeClassifierGNN_conv(nn.Module):
    def __init__(self,
                 edge_in_dim: int = 154,
                 hidden: int      = 32,
                 n_layers: int    = 2,
                 dropout: float   = 0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # input size of node feature is now 33*2+3 (from node_lbls)
        in_channels = 6*2+3
        for _ in range(n_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden,  in_channels * hidden)  # weight matrix
            )
            conv = NNConv(in_channels, hidden, edge_nn, aggr='mean')
            self.edge_mlps.append(edge_nn)
            self.convs.append(conv)
            self.dropouts.append(nn.Dropout(dropout))
            in_channels = hidden

        self.lin = nn.Linear(hidden, 1)  # binary output

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv, dropout in zip(self.convs, self.dropouts):
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = dropout(x)
        out = self.lin(x).squeeze(-1)           # (num_nodes,)
        return out                              # raw logits


class NodeClassifierGNN_gat(nn.Module):
    def __init__(self,
                 in_channels: int,  # Change from edge_in_dim
                 hidden: int      = 32,
                 n_layers: int    = 2,
                 dropout: float   = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n_layers):
            # GATConv doesn't use an edge MLP like NNConv
            # It takes node feature dimensions as input/output
            conv = GATConv(in_channels, hidden, heads=4) # Using 4 attention heads is a good start
            self.convs.append(conv)
            self.dropouts.append(nn.Dropout(dropout))
            in_channels = hidden * 4 # The output dimension is hidden_dim * heads

        # Adjust the final linear layer for the new input size
        self.lin = nn.Linear(hidden * 4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index # GATConv doesn't need edge_attr by default
        for conv, dropout in zip(self.convs, self.dropouts):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = dropout(x)
        out = self.lin(x).squeeze(-1)
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N,) or (N,1), targets: (N,) or (N,1)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = torch.where(targets == 1, probs, 1 - probs).clamp(1e-6, 1 - 1e-6)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train_model(train_graphs, val_graphs, test_graph, model, device, num_epochs=100, patience=10, lr=1e-4, weight_decay=1e-3, train_batch_size=1, val_batch_size=1):
    train_loader = DataLoader(train_graphs, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=val_batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Focal loss for imbalance
    # all_labels = np.concatenate([data.y.cpu().numpy() for data in train_graphs])
    # all_labels = all_labels[all_labels >= 0]
    # # Calculate alpha as the proportion of the dataset that is the negative class
    # alpha = np.sum(all_labels == 0) / len(all_labels)

    # Now use this calculated alpha
    # criterion = FocalLoss(alpha=alpha, gamma=.5, reduction='mean')
    
    # ── binary cross entropy ──────────────────────────────────────────────────────
    all_labels = np.concatenate([data.y.cpu().numpy() for data in train_graphs])
    all_labels = all_labels[all_labels >= 0] # Filter out unknowns
    if np.sum(all_labels == 1) > 0:
        num_pos = np.sum(all_labels == 1)
        num_neg = np.sum(all_labels == 0)
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device) # Default if no positives in batch
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap = 0
    best_f1 = 0
    best_loss = np.inf
    epochs_no_improve = 0
    best_state_dict = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.
        for data in train_loader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")
        # Early stopping: evaluate on validation set every epoch
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                labels = data.y.cpu().numpy()
                mask = labels >= 0
                val_probs.append(probs[mask])
                val_labels.append(labels[mask])
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        if len(np.unique(val_labels)) > 1:
            val_ap = average_precision_score(val_labels, val_probs)
            pred_bin = (val_probs >= 0.5).astype(int)
            _, _, val_f1, _ = precision_recall_fscore_support(val_labels, pred_bin, average='binary', zero_division=0)
            if val_ap > best_ap:
                best_ap = val_ap
                best_f1 = val_f1
                best_loss = total_loss / len(train_loader)
                best_state_dict = copy.deepcopy(model.state_dict())
                print(colored(f"Val AP {val_ap:.3f} (F1={val_f1:.3f}) >  Best Val AP {best_ap:.3f} (F1={best_f1:.3f})", 'green'))
                epochs_no_improve = 0
            else:
                print(colored(f"Val AP {val_ap:.3f} (F1={val_f1:.3f}) <= Best Val AP {best_ap:.3f} (F1={best_f1:.3f})", 'red'))
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(colored(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)", 'yellow'))
                print(colored(f"Best loss={best_loss:.3f}, Current loss={total_loss / len(train_loader)}", 'red'))
                break
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                data = test_graph.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                labels = data.y.cpu().numpy()
                mask = labels >= 0
                if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
                    continue
                auc = roc_auc_score(labels[mask], probs[mask])
                ap = average_precision_score(labels[mask], probs[mask])
                pred_bin = (probs[mask] >= 0.5).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(labels[mask], pred_bin, average='binary', zero_division=0)
                pid = test_graph.pid.cpu().numpy()[0]
                print(colored(f"TEST pid={pid}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}, prec={precision:.3f}, recall={recall:.3f}", 'blue'))
        else:
            print("Not enough class variety in validation set for metrics. Stopping.")
            break
    # Restore best model before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def loo_crossval(graphs, device, edge_in_dim, hidden=32, n_layers=2, dropout=0.2, num_epochs=100, patience=10, lr=1e-4, weight_decay=1e-3, train_batch_size=1, val_batch_size=1):
    results = []
    for i in range(len(graphs)):
        test_graph = graphs[i]
        train_graphs = graphs[:i] + graphs[i+1:]
        # Split train_graphs into train/val for early stopping
        if len(train_graphs) > 1:
            train_fold_graphs, val_fold_graphs = train_test_split(train_graphs, test_size=0.1, random_state=42)
            # If val set is empty (rare, e.g. very small data), use 1 for val
            if len(val_fold_graphs) == 0:
                val_fold_graphs = train_fold_graphs[-1:]
                train_fold_graphs = train_fold_graphs[:-1]
        else:
            train_fold_graphs = train_graphs
            val_fold_graphs = train_graphs
        model = NodeClassifierGNN_conv(edge_in_dim=edge_in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)
        # node_feature_dim = len(targets)*2 + 3 # Calculate this based on your new, smaller targets list
        # model = NodeClassifierGNN_gat(in_channels=node_feature_dim, hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)

        model = train_model(train_fold_graphs, val_fold_graphs, test_graph,model, device, num_epochs=num_epochs, patience=patience, lr=lr, weight_decay=weight_decay, train_batch_size=train_batch_size, val_batch_size=val_batch_size)
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(val_fold_graphs, batch_size=val_batch_size, shuffle=False)
            val_probs = []
            val_labels = []
            for data in val_loader:
                data = data.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                labels = data.y.cpu().numpy()
                mask = labels >= 0
                val_probs.append(probs[mask])
                val_labels.append(labels[mask])
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)

        # Search for the threshold that maximizes your target metric (e.g., F0.5-score)
        thresholds = np.linspace(0.01, 0.99, 100)
        best_f_beta = -1
        best_thresh = 0.5
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            f_beta = fbeta_score(val_labels, preds, beta=0.5, zero_division=0)
            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_thresh = thresh
        print(f"Optimal threshold found: {best_thresh:.3f} with F0.5-score: {best_f_beta:.3f}")
        # Evaluate on left-out graph
        model.eval()
        with torch.no_grad():
            data = test_graph.to(device)
            probs = torch.sigmoid(model(data)).cpu().numpy()
            labels = data.y.cpu().numpy()
            mask = labels >= 0
            if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
                continue
            auc = roc_auc_score(labels[mask], probs[mask])
            ap = average_precision_score(labels[mask], probs[mask])
            pred_bin = (probs[mask] >= best_thresh).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels[mask], pred_bin, average='binary', zero_division=0)
            results.append({
                'auc': auc,
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        pid = test_graph.pid.cpu().numpy()[0]
        print(f"LOO {i+1}/{len(graphs)} pid={pid}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}, prec={precision:.3f}, recall={recall:.3f}")
        break
    return results

def evaluate_graphs(graphs, model, device, set_name="Test"):
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            probs = torch.sigmoid(model(data)).cpu().numpy()  # (num_nodes,)
            labels = data.y.cpu().numpy()                     # (num_nodes,)
            mask = labels >= 0                                # ignore unknown labels
            all_probs.append(probs[mask])
            all_labels.append(labels[mask])
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    if len(np.unique(all_labels)) < 2:
        print(f"Not enough positive/negative samples in {set_name} set for ROC/AUC.")
        return None
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    # Choose threshold 0.5 for confusion matrix and F1/precision/recall
    pred_bin = (all_probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_bin, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, pred_bin)
    print(f"\n{set_name} set metrics:")
    print(f"AUC: {auc:.3f}")
    print(f"Average Precision (AP): {ap:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    # plt.plot(fpr, tpr, label=f'{set_name} ROC (AUC={auc:.2f})')
    return fpr, tpr, auc

if __name__ == "__main__":
    # For reproducibility
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hardcoded hyperparameters
    hidden = 64
    n_layers = 3
    dropout = 0.4
    num_epochs = 100
    patience = 10
    lr = 5e-5    
    weight_decay = 1e-4
    train_batch_size = 1
    val_batch_size = 1

    skip = ["pid","electrode_pair","electrode_a","electrode_b",
            "soz_a","soz_b","ilae","electrode_pair_names",
            "electrode_a_name","electrode_b_name","miccai_label_a",
            "miccai_label_b","age_days","age_years","soz_bin","soz_sum","etiology"]

    if os.path.exists('patient_graphs.pt'):
        data = torch.load('patient_graphs.pt', weights_only=False)
        graphs = data['graphs']
        targets = data['targets']
        print("Loaded graphs from disk.")
    else:
        graphs, targets = build_patient_graphs("/media/dan/Data/data/renamed_mean_data.csv", skip)
        torch.save({'graphs': graphs, 'targets': targets}, 'patient_graphs.pt')
        print("Processed and saved graphs.")

    # Filter to only patients with ilae == 1
    filtered_graphs = [g for g in graphs if g.ilae.item() == 1]
    # filtered_graphs = graphs
    print(f"Total ilae==1 patients: {len(filtered_graphs)}")

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = loo_crossval(
        filtered_graphs, device, edge_in_dim=len(targets),
        hidden=hidden, n_layers=n_layers, dropout=dropout,
        num_epochs=num_epochs, patience=patience,
        lr=lr, weight_decay=weight_decay,
        train_batch_size=train_batch_size, val_batch_size=val_batch_size
    )
    if results:
        aucs = [r['auc'] for r in results]
        aps = [r['ap'] for r in results]
        f1s = [r['f1'] for r in results]
        print(f"\nLOO-CV Results (n={len(results)}):")
        print(f"Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"Mean AP: {np.mean(aps):.3f} ± {np.std(aps):.3f}")
        print(f"Mean F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    else:
        print("No valid LOO splits for metrics.")

    print("\nParameters used:")
    print(f"  hidden: {hidden}")
    print(f"  n_layers: {n_layers}")
    print(f"  dropout: {dropout}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  patience: {patience}")
    print(f"  lr: {lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  train_batch_size: {train_batch_size}")
    print(f"  val_batch_size: {val_batch_size}")
           

