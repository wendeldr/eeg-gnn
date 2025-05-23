import os
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv # Example GNN layer
from sklearn.model_selection import train_test_split # For splitting PIDs

# --- I. Configuration & Paths ---
EDF_BASE_PATH = "/media/dan/Data/data/baseline_patients/baseline_edfs"
CONTACT_INFO_PATH = 'contact_info.csv'
SFREQ_TARGET = 2048
SEGMENT_LENGTH = 1024 # e.g., (4096) = 2 seconds at 2048Hz , (1024) = 0.5 seconds at 2048Hz
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- II. Data Pipeline Components ---

class PatientEEGDataset(Dataset):
    def __init__(self, contact_info_df, edf_base_path, patient_ids, target_sfreq):
        self.contact_info_df = contact_info_df
        self.edf_base_path = edf_base_path
        self.patient_ids = [pid for pid in patient_ids if self._check_patient_sfreq(pid, target_sfreq)]
        self.target_sfreq = target_sfreq
        print(f"Initialized Dataset with {len(self.patient_ids)} valid patients.")

    def _check_patient_sfreq(self, pid, target_sfreq):
        edf_file = os.path.join(self.edf_base_path, f"{int(pid):03}_Baseline.EDF")
        if not os.path.exists(edf_file):
            print(f"Warning: EDF file not found for pid {pid}. Skipping.")
            return False
        try:
            raw_info = mne.io.read_raw_edf(edf_file, preload=False, verbose=False).info
            if raw_info['sfreq'] != target_sfreq:
                print(f"Skipping patient {pid}: sfreq {raw_info['sfreq']} != {target_sfreq}")
                return False
            # Check if patient has contacts in contact_info_df
            if not contact_info_df[contact_info_df['pid'] == pid].shape[0] > 0:
                print(f"Skipping patient {pid}: No contacts found in contact_info_df.")
                return False
            return True
        except Exception as e:
            print(f"Error reading info for patient {pid}: {e}. Skipping.")
            return False


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        
        patient_contacts_df = self.contact_info_df[self.contact_info_df['pid'] == pid].copy()
        
        edf_file = os.path.join(self.edf_base_path, f"{int(pid):03}_Baseline.EDF")
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False) # Preload data here

        # Filter channels based on contact_info_df for this patient
        # Ensure EEG prefix and exact match
        raw_ch_names_no_prefix = [ch.replace('EEG ', '').strip() for ch in raw.ch_names if 'EEG ' in ch]
        patient_contacts_df['eeg_ch_name_match'] = patient_contacts_df['electrode'].apply(lambda x: x.strip())
        
        valid_contacts_df = patient_contacts_df[patient_contacts_df['eeg_ch_name_match'].isin(raw_ch_names_no_prefix)]
        
        # Create a mapping from original contact name to EEG channel name for picking
        ch_picks_map = {row['eeg_ch_name_match']: f"EEG {row['electrode'].strip()}" for _, row in valid_contacts_df.iterrows()}
        ordered_picks = [ch_picks_map[name] for name in valid_contacts_df['eeg_ch_name_match']] # Pick in order of valid_contacts_df

        if not ordered_picks:
            print(f"Warning: No valid channels to pick for patient {pid} after matching. Returning None.")
            # This sample will be filtered out by the collate_fn or needs error handling
            return None 

        raw_eeg_data = raw.get_data(picks=ordered_picks) # (num_contacts, time_points)
        
        node_spatial_feat = valid_contacts_df[['X_recoded_dl2', 'Y_recoded_dl2', 'Z_recoded_dl2']].values
        soz_labels = valid_contacts_df['soz'].values
        
        return raw_eeg_data, node_spatial_feat, soz_labels, pid


class CustomCollate:
    def __init__(self, segment_length):
        self.segment_length = segment_length

    def __call__(self, batch):
        # Filter out None items from batch (e.g., if a patient had no valid channels)
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None, None # Or raise an error

        pyg_data_list = []
        all_epd_segments = []
        all_relative_spatial = []

        for raw_eeg_data, node_spatial_feat, soz_labels, _ in batch: # pid not used in collate
            num_contacts, total_time_points = raw_eeg_data.shape

            if num_contacts < 2: # Need at least 2 contacts for edges
                continue

            # Edge Index (fully connected upper triangle)
            adj = np.ones((num_contacts, num_contacts))
            adj_upper = np.triu(adj, k=1)
            edge_src, edge_dst = np.where(adj_upper == 1)
            
            if edge_src.size == 0: # No edges if only 1 contact somehow passed
                continue

            edge_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.long)
            num_edges = edge_index.shape[1]

            patient_epd_segments = []
            patient_relative_spatial = []

            for k in range(num_edges):
                u, v = edge_src[k], edge_dst[k]
                epd_full = raw_eeg_data[u, :] - raw_eeg_data[v, :]

                if total_time_points > self.segment_length:
                    start = np.random.randint(0, total_time_points - self.segment_length + 1)
                    epd_segment = epd_full[start : start + self.segment_length]
                else:
                    epd_segment = np.pad(epd_full, (0, self.segment_length - total_time_points), 'constant')
                
                patient_epd_segments.append(torch.tensor(epd_segment, dtype=torch.float32).unsqueeze(0)) # (1, segment_length)

                rel_spatial = node_spatial_feat[u, :] - node_spatial_feat[v, :]
                patient_relative_spatial.append(torch.tensor(rel_spatial, dtype=torch.float32))
            
            if not patient_epd_segments: # If no edges were processed
                continue
                
            all_epd_segments.extend(patient_epd_segments)
            all_relative_spatial.extend(patient_relative_spatial)

            x = torch.tensor(node_spatial_feat, dtype=torch.float32)
            y = torch.tensor(soz_labels, dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_contacts)
            pyg_data_list.append(data)

        if not pyg_data_list:
            return None, None, None

        pyg_batch = Batch.from_data_list(pyg_data_list)
        batched_epd_segments = torch.stack(all_epd_segments, dim=0) # (total_edges_in_batch, 1, segment_length)
        batched_rel_spatial = torch.stack(all_relative_spatial, dim=0) # (total_edges_in_batch, 3)
        
        return pyg_batch, batched_epd_segments, batched_rel_spatial


# --- III. Model Components ---

class EPDFeatureExtractorCNN(nn.Module):
    def __init__(self, input_channels=1, num_features_out=64, segment_length=SEGMENT_LENGTH):
        super().__init__()
        # Example architecture - this needs careful tuning based on segment_length!
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=64, stride=8, padding=28), # (L_in - K + 2P)/S + 1
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Halves length

            nn.Conv1d(16, 32, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size dynamically based on segment_length
        # Pass a dummy input through conv_layers to get the output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, segment_length)
            cnn_out_shape = self.conv_layers(dummy_input).shape
            flattened_size = cnn_out_shape[1] * cnn_out_shape[2] # C_out * L_out_cnn

        self.fc = nn.Linear(flattened_size, num_features_out)
        print(f"EPD CNN initialized. Conv output flattened size: {flattened_size}, FC output: {num_features_out}")


    def forward(self, x): # x: (batch_edges, 1, segment_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x


class PatientGNN(nn.Module):
    def __init__(self, node_feat_dim=3, epd_cnn_feat_out=64, rel_spatial_dim=3, 
                 gnn_hidden_dim=128, num_gnn_layers=2, heads_gat=4, dropout_rate=0.3): # Reduced dropout
        super().__init__()
        self.epd_cnn_feat_out = epd_cnn_feat_out
        self.rel_spatial_dim = rel_spatial_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        self.epd_cnn = EPDFeatureExtractorCNN(num_features_out=epd_cnn_feat_out, segment_length=SEGMENT_LENGTH) # SEGMENT_LENGTH global for now

        # Node feature projection (optional, can directly use XYZ if normalized)
        self.node_proj = nn.Linear(node_feat_dim, gnn_hidden_dim)

        # Edge feature MLP
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(epd_cnn_feat_out + rel_spatial_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim) # Output dim matches GATConv edge_dim
        )

        self.gnn_layers = nn.ModuleList()
        current_node_dim = gnn_hidden_dim # After node_proj
        for i in range(num_gnn_layers):
            if i == num_gnn_layers - 1: # Last layer
                self.gnn_layers.append(
                    GATConv(current_node_dim, gnn_hidden_dim, heads=1, concat=False, # Single head, no concat
                            dropout=dropout_rate, edge_dim=gnn_hidden_dim) 
                )
                current_node_dim = gnn_hidden_dim
            else:
                self.gnn_layers.append(
                    GATConv(current_node_dim, gnn_hidden_dim, heads=heads_gat, concat=True,
                            dropout=dropout_rate, edge_dim=gnn_hidden_dim)
                )
                current_node_dim = gnn_hidden_dim * heads_gat
        
        self.classifier = nn.Sequential(
            nn.Linear(current_node_dim, gnn_hidden_dim // 2), # current_node_dim is output of last GNN
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )
        print("PatientGNN initialized.")

    def forward(self, pyg_batch, epd_segments, rel_spatial_features):
        # 1. EPD features from CNN
        # epd_segments: (total_edges_in_batch, 1, segment_length)
        learned_epd_feats = self.epd_cnn(epd_segments) # (total_edges, epd_cnn_feat_out)

        # 2. Combine with relative spatial features for edges
        # rel_spatial_features: (total_edges, rel_spatial_dim)
        combined_edge_input = torch.cat([learned_epd_feats, rel_spatial_features], dim=-1)
        final_edge_attr = self.edge_feature_mlp(combined_edge_input) # (total_edges, gnn_hidden_dim)

        # 3. Node features
        x = pyg_batch.x # (total_nodes_in_batch, node_feat_dim)
        x = self.node_proj(x) # (total_nodes, gnn_hidden_dim)
        
        edge_index = pyg_batch.edge_index

        # 4. GNN propagation
        for i, l in enumerate(self.gnn_layers):
            x = l(x, edge_index, edge_attr=final_edge_attr)
            if i < len(self.gnn_layers) -1 : # Apply ReLU and dropout except for last GNN layer output
                 x = F.relu(x)
                 x = F.dropout(x, p=0.3, training=self.training) # Use the same dropout as in GATConv for consistency

        # 5. Classification
        node_predictions = self.classifier(x) # (total_nodes, 1)
        return node_predictions


# --- IV. Training Setup ---
if __name__ == '__main__': # Ensure this runs only when script is executed directly
    contact_info_df = pd.read_csv(CONTACT_INFO_PATH)
    all_pids = contact_info_df['pid'].unique()

    tmp_dataset = PatientEEGDataset(contact_info_df, EDF_BASE_PATH, all_pids, target_sfreq=SFREQ_TARGET)
    all_pids = tmp_dataset.patient_ids
    
    # Basic split - consider more robust (e.g., group shuffle split if multiple records per pid affect splitting)
    train_pids, test_pids = train_test_split(all_pids, test_size=0.3, random_state=42)
    val_pids, test_pids = train_test_split(test_pids, test_size=0.5, random_state=42) # test is 0.15, val is 0.15

    print(f"Train PIDs: {len(train_pids)}, Val PIDs: {len(val_pids)}, Test PIDs: {len(test_pids)}")

    train_dataset = PatientEEGDataset(contact_info_df, EDF_BASE_PATH, train_pids, target_sfreq=SFREQ_TARGET)
    val_dataset = PatientEEGDataset(contact_info_df, EDF_BASE_PATH, val_pids, target_sfreq=SFREQ_TARGET)

    collate_fn = CustomCollate(segment_length=SEGMENT_LENGTH)

    # WARNING: batch_size needs to be small due to GPU memory. Start with 1.
    # num_workers > 0 can speed up data loading but uses more RAM. Start with 0 for debugging.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True if DEVICE.type == 'cuda' else False)

    # Model Hyperparameters (example)
    NODE_FEAT_DIM = 3 # X,Y,Z
    EPD_CNN_FEAT_OUT = 64
    REL_SPATIAL_DIM = 3
    GNN_HIDDEN_DIM = 128 # Increased GNN hidden dim
    NUM_GNN_LAYERS = 2
    GAT_HEADS = 4 # Number of attention heads for GATConv
    DROPOUT_RATE = 0.3

    model = PatientGNN(
        node_feat_dim=NODE_FEAT_DIM,
        epd_cnn_feat_out=EPD_CNN_FEAT_OUT,
        rel_spatial_dim=REL_SPATIAL_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        heads_gat=GAT_HEADS,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss() # Handles sigmoid
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # AdamW and weight decay

    # --- V. Training Loop ---
    NUM_EPOCHS = 20 # Start with a small number for testing

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_batches = 0
        for pyg_batch, epd_segments, rel_spatial_features in train_loader:
            if pyg_batch is None: # Skip if collate_fn returned None due to empty batch
                continue
                
            pyg_batch = pyg_batch.to(DEVICE)
            epd_segments = epd_segments.to(DEVICE)
            rel_spatial_features = rel_spatial_features.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pyg_batch, epd_segments, rel_spatial_features)
            
            # Ensure target y is also on device and has correct shape
            targets = pyg_batch.y.unsqueeze(1).to(DEVICE)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for pyg_batch, epd_segments, rel_spatial_features in val_loader:
                if pyg_batch is None:
                    continue
                pyg_batch = pyg_batch.to(DEVICE)
                epd_segments = epd_segments.to(DEVICE)
                rel_spatial_features = rel_spatial_features.to(DEVICE)

                outputs = model(pyg_batch, epd_segments, rel_spatial_features)
                targets = pyg_batch.y.unsqueeze(1).to(DEVICE)
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item()
                val_batches += 1
                
                # Store predictions and targets for metrics
                all_preds.append(torch.sigmoid(outputs).cpu()) # Apply sigmoid for probability
                all_targets.append(targets.cpu())
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Val Loss: {avg_val_loss:.4f}")

        if all_preds:
            all_preds_cat = torch.cat(all_preds, dim=0).squeeze().numpy()
            all_targets_cat = torch.cat(all_targets, dim=0).squeeze().numpy()
            # Example: Calculate F1 score (you'll need sklearn for this)
            # from sklearn.metrics import f1_score, roc_auc_score
            # f1 = f1_score(all_targets_cat, all_preds_cat > 0.5) # Threshold at 0.5
            # auc = roc_auc_score(all_targets_cat, all_preds_cat)
            # print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Val F1: {f1:.4f}, Val AUC: {auc:.4f}")
        print("-" * 30)