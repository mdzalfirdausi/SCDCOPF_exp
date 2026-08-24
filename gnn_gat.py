import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv

# =============================================================================
# 1. GRAPH ATTENTION NETWORK (GAT) ARCHITECTURE
# =============================================================================
class ContingencyGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_contingencies, heads=4):
        super(ContingencyGAT, self).__init__()
        # Graph Attention Layers mapping complex grid topologies
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        
        # Classification Head: Outputs [num_buses, num_contingencies]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2, num_contingencies) 
        )

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.leaky_relu(h, 0.1)
        h = self.gat2(h, edge_index)
        h = F.leaky_relu(h, 0.1)
        
        logits = self.classifier(h)
        return logits

# =============================================================================
# 2. GRAPH DATASET BUILDER (PANDAS CSV INTEGRATION)
# =============================================================================
def build_graph_dataset(case_path, load_csv_path, labels_csv_path, baseMVA=100.0):
    print(f"Building PyTorch Geometric Dataset from {load_csv_path}...")
    
    # 1. Load Topology from Excel
    case = pd.read_excel(case_path, sheet_name=['bus', 'gen', 'branch'])
    bus_df, gen_df, branch_df = case['bus'], case['gen'], case['branch']
    
    num_buses = len(bus_df)
    num_contingencies = len(gen_df)
    
    # 2. Map Edges (0-indexed for PyTorch Geometric edge_index)
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_df['bus_i'].values)}
    edge_source = [bus_idx_map[i] for i in branch_df['bus_i'].values]
    edge_target = [bus_idx_map[j] for j in branch_df['bus_j'].values]
    
    # Undirected graph: Add reverse edges
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)
    
    # 3. Static Node Features & Generator Masks
    pmax_feat = np.zeros(num_buses)
    pmin_feat = np.zeros(num_buses)
    gen_mask = torch.zeros(num_buses, dtype=torch.bool)
    
    gen_bus_indices = []
    for _, row in gen_df.iterrows():
        b_idx = bus_idx_map[row['bus_i']]
        pmax_feat[b_idx] = row['Pmax'] / baseMVA
        pmin_feat[b_idx] = row['Pmin'] / baseMVA
        gen_mask[b_idx] = True
        gen_bus_indices.append(b_idx)
        
    # 4. Load Scenarios and Optimal Binaries
    load_profiles = pd.read_csv(load_csv_path)
    target_binaries = pd.read_csv(labels_csv_path).values 
    
    dataset = []
    for s in range(len(load_profiles)):
        # Node Feature X: [Pd, Pmax, Pmin]
        pd_feat = load_profiles.iloc[s].values[:num_buses] / baseMVA
        x_tensor = torch.tensor(np.column_stack((pd_feat, pmax_feat, pmin_feat)), dtype=torch.float32)
        
        # Format y_raw into [num_buses, num_contingencies] node-level targets
        y_raw = target_binaries[s].reshape(num_contingencies, num_contingencies) 
        y_tensor = torch.zeros((num_buses, num_contingencies), dtype=torch.float32)
        
        for i, b_idx in enumerate(gen_bus_indices):
            y_tensor[b_idx, :] = torch.tensor(y_raw[:, i], dtype=torch.float32)

        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, gen_mask=gen_mask)
        dataset.append(data)
        
    print(f"Successfully built {len(dataset)} graph scenarios!")
    return dataset, num_contingencies

# =============================================================================
# 3. TRAINING LOOP
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='pglib_opf_case30_ieee')
    args = parser.parse_args()
    
    case_path = f'../excel_outputs/{args.case}.xlsx'
    load_csv = f'data/{args.case}_generated_loads.csv'
    labels_csv = f'data/{args.case}_optimal_binaries.csv' # Offline Ground Truth
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deploying workload to device: {device}")
    
    dataset, num_kg = build_graph_dataset(case_path, load_csv, labels_csv)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ContingencyGAT(in_channels=3, hidden_channels=128, num_contingencies=num_kg).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nInitiating GAT Training...")
    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index)
            
            # Mask out load buses to calculate BCE loss only for Generator Buses
            pred_gens = logits[batch.gen_mask].view(-1)
            target_gens = batch.y[batch.gen_mask].view(-1)
            
            loss = criterion(pred_gens, target_gens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | BCE Loss: {total_loss / len(dataset):.6f}")
            
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), f"data/models/{args.case}_gat_binary_mapper.pth")
    print("\nTraining Complete. GAT Model Saved.")