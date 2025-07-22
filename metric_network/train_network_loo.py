import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool  # we won’t pool, but handy later


def build_patient_graphs(csv_path: str,
                         skip_cols: list[str]) -> tuple[list[Data], list[str]]:

    df = pd.read_csv(csv_path, low_memory=False)

    # remove any patients with only one value of soz_sum
    df = df[df.groupby('pid')['soz_sum'].transform('nunique') > 1]
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

    targets = [c for c in df.columns if c not in skip_cols]

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
        edges, e_attr = [], []

        y = torch.full((n_nodes,), -1, dtype=torch.long)

        node_lbls = torch.full((n_nodes,3), -1, dtype=torch.long)
        # Collect edges and their target (154‑D) attributes
        for _, row in grp.iterrows():
            i, j = int(row.electrode_a) - 1, int(row.electrode_b) - 1
            feat = row[targets].to_numpy(dtype=np.float32)

            edges.append([i, j])
            e_attr.append(feat)
            # edges += [[i, j], [j, i]]
            # e_attr += [feat, feat]

            # Node labels (only need to be set once)
            if y[i] == -1:
                y[i] = int(row.soz_a)
                node_lbls[i,:] = torch.tensor([row.miccai_label_a_idx, row.age_days, row.etiology_idx])
                
            if y[j] == -1:
                y[j] = int(row.soz_b)
                node_lbls[j,:] = torch.tensor([row.miccai_label_b_idx, row.age_days, row.etiology_idx])
            
        edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(e_attr,dtype=torch.float32).contiguous()

        # Very simple node feature: constant 1 (shape [n_nodes, 1])
        x = torch.ones((n_nodes, 1), dtype=torch.float32)

        ilae_label = torch.tensor([grp.ilae.values[0]], dtype=torch.float32)

        data_list.append(Data(x=x,
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              y=y,
                              pid=torch.tensor([pid]),
                              ilae=ilae_label))   # keep pid if you need it

    return data_list, targets



class NodeClassifierGNN(nn.Module):
    def __init__(self,
                 edge_in_dim: int = 154,
                 hidden: int      = 32,
                 n_layers: int    = 2,
                 dropout: float   = 0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # input size of node feature is 1 (used constant ones)
        in_channels = 1
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


def train_model(train_graphs, model, device, num_epochs=100, patience=10):
    loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    # basic bce loss
    # criterion = nn.BCEWithLogitsLoss()

    # weighted bce loss
    all_labels = np.concatenate([data.y.cpu().numpy() for data in train_graphs])
    num_pos = np.sum(all_labels == 1)
    num_neg = np.sum(all_labels == 0)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0
    epochs_no_improve = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.
        for data in loader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        # Early stopping: evaluate on train set every epoch
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for data in train_graphs:
                data = data.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                labels = data.y.cpu().numpy()
                mask = labels >= 0
                all_probs.append(probs[mask])
                all_labels.append(labels[mask])
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        if len(np.unique(all_labels)) > 1:
            train_auc = roc_auc_score(all_labels, all_probs)
            if train_auc > best_auc:
                best_auc = train_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        else:
            break
    return model


def loo_crossval(graphs, device, edge_in_dim, hidden=32, n_layers=2, dropout=0.2, num_epochs=100, patience=10):
    results = []
    for i in range(len(graphs)):
        test_graph = graphs[i]
        train_graphs = graphs[:i] + graphs[i+1:]
        model = NodeClassifierGNN(edge_in_dim=edge_in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)
        model = train_model(train_graphs, model, device, num_epochs=num_epochs, patience=patience)
        # Evaluate on left-out graph
        model.eval()
        with torch.no_grad():
            data = test_graph.to(device)
            probs = torch.sigmoid(model(data)).cpu().numpy()
            labels = data.y.cpu().numpy()
            mask = labels >= 0
            if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
                # Not enough data for metrics
                continue
            auc = roc_auc_score(labels[mask], probs[mask])
            ap = average_precision_score(labels[mask], probs[mask])
            pred_bin = (probs[mask] >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels[mask], pred_bin, average='binary', zero_division=0)
            results.append({
                'auc': auc,
                'ap': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        pid = test_graph.pid.cpu().numpy()[0]
        print(f"LOO {i+1}/{len(graphs)} pid={pid}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}")
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
    # filtered_graphs = [g for g in graphs if g.ilae.item() >= 1]
    filtered_graphs = graphs
    print(f"Total ilae==1 patients: {len(filtered_graphs)}")

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = loo_crossval(filtered_graphs, device, edge_in_dim=len(targets), hidden=32, n_layers=2, dropout=0.2, num_epochs=100, patience=10)
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


