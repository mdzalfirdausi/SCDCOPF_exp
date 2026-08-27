import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import argparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# ==========================================
# 1. DATA PREPARATION (ZONAL & PYG)
# ==========================================
def create_zonal_data(case, zone1_buses, zone2_buses):
    """Splits the grid into zones."""
    zonal_data = {'zone1': {}, 'zone2': {}, 'tie_lines': [], 'boundary_buses': [], 'global_Kg': []}
    zonal_data['global_Kg'] = [int(x) for x in case['gen']['gen_ID'].tolist()]
    
    branch_df = case['branch']
    boundary_buses = set()
    
    for idx, row in branch_df.iterrows():
        f_bus = int(row['bus_i'])
        t_bus = int(row['bus_j'])
        if (f_bus in zone1_buses and t_bus in zone2_buses) or (f_bus in zone2_buses and t_bus in zone1_buses):
            zonal_data['tie_lines'].append(int(row['line_ID']))
            boundary_buses.add(f_bus)
            boundary_buses.add(t_bus)
            
    zonal_data['boundary_buses'] = sorted(list(boundary_buses))
            
    zonal_data['zone1']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone1']['gen']['gen_ID'])].copy()
    zonal_data['zone1']['branch'] = branch_df[(branch_df['bus_i'].isin(zone1_buses)) & (branch_df['bus_j'].isin(zone1_buses))].copy()
    
    zonal_data['zone2']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone2']['gen']['gen_ID'])].copy()
    zonal_data['zone2']['branch'] = branch_df[(branch_df['bus_i'].isin(zone2_buses)) & (branch_df['bus_j'].isin(zone2_buses))].copy()
    
    for z in ['zone1', 'zone2']:
        zonal_data[z]['baseMVA'] = case['baseMVA']
        zonal_data[z]['global_Kg'] = zonal_data['global_Kg']
    
    return zonal_data

def create_pyg_dataset(zonal_data, load_data_np, baseMVA):
    """Converts Pandas zonal data and load matrices into PyTorch Geometric graphs."""
    bus_df = zonal_data['bus']
    gen_df = zonal_data['gen']
    branch_df = zonal_data['branch']
    
    num_buses = len(bus_df)
    
    # Map bus IDs to continuous indices 0...N-1
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_df['bus_i'].values)}
    
    # Create edge_index (Topology)
    edge_source = [bus_idx_map[i] for i in branch_df['bus_i'].values if i in bus_idx_map]
    edge_target = [bus_idx_map[j] for j in branch_df['bus_j'].values if j in bus_idx_map]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)
    
    # Create Generator Masks and Limits
    gen_mask = torch.zeros(num_buses, dtype=torch.bool)
    Pmax = torch.zeros(num_buses, dtype=torch.float32)
    Pmin = torch.zeros(num_buses, dtype=torch.float32)
    
    for _, row in gen_df.iterrows():
        if row['bus_i'] in bus_idx_map:
            b_idx = bus_idx_map[row['bus_i']]
            gen_mask[b_idx] = True
            Pmax[b_idx] = row['Pmax'] / baseMVA
            Pmin[b_idx] = row['Pmin'] / baseMVA

    # Build Graph Scenarios
    dataset = []
    for s in range(len(load_data_np)):
        Pd = torch.tensor(load_data_np[s], dtype=torch.float32).view(-1, 1)
        # Node features: [Pd, Pmax, Pmin]
        x = torch.cat([Pd, Pmax.view(-1, 1), Pmin.view(-1, 1)], dim=1)
        
        data = Data(x=x, edge_index=edge_index)
        data.Pd = Pd.view(-1)
        data.gen_mask = gen_mask
        data.Pmax = Pmax
        data.Pmin = Pmin
        dataset.append(data)
        
    return dataset

# ==========================================
# 2. GRAPH NEURAL NETWORK ARCHITECTURE
# ==========================================
class Zone_ADMM_GNN(nn.Module):
    def __init__(self, num_boundaries, num_global_kg):
        super(Zone_ADMM_GNN, self).__init__()
        
        self.num_kg_base = num_global_kg + 1 
        self.num_boundaries = num_boundaries
        
        # Graph Attention Layers (Node processing)
        self.gat1 = GATConv(3, 64, heads=4, concat=True)
        self.gat2 = GATConv(64 * 4, 128, heads=1, concat=False)
        
        # Generator Head (Node-level continuous output)
        self.pg_head = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Linear(64, 1)
        )
        
        # Phase Angle Head (Graph-level continuous output for boundary consensus)
        self.va_head = nn.Sequential(
            nn.Linear(128 + (self.num_kg_base * num_boundaries * 2), 128), 
            nn.LeakyReLU(0.1), 
            nn.Linear(128, self.num_kg_base * num_boundaries)
        )
        
        # Contingency ZK Head (Graph-level BERNOULLI PROBABILITIES)
        self.zk_head = nn.Sequential(
            nn.Linear(128 + (num_global_kg * 2), 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_global_kg),
            nn.Sigmoid() # Erdős formulation: Outputs probabilities between 0 and 1
        )

    def forward(self, data, Va_target, u_va, zk_target, u_zk):
        x, edge_index = data.x, data.edge_index
        
        # Safely handle batch attribute and num_graphs for single graph inference
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            num_graphs = getattr(data, 'num_graphs', batch.max().item() + 1)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            num_graphs = 1
        
        # 1. Message Passing (Graph Topology)
        h = F.leaky_relu(self.gat1(x, edge_index), 0.1)
        h = F.leaky_relu(self.gat2(h, edge_index), 0.1)
        
        # 2. Predict Pg (Continuous Generators)
        h_gen = h[data.gen_mask]
        raw_pg = self.pg_head(h_gen).squeeze(-1)
        Pg_base = torch.sigmoid(raw_pg) * (data.Pmax[data.gen_mask] - data.Pmin[data.gen_mask]) + data.Pmin[data.gen_mask]
        
        h_global = global_mean_pool(h, batch)
        
        # 3. Predict Va (Graph-level extraction for phase angles)
        va_t_flat = Va_target.view(num_graphs, -1)
        uva_flat = u_va.view(num_graphs, -1) / 10000.0
        
        h_global_va = torch.cat([h_global, va_t_flat, uva_flat], dim=1)
        
        raw_va = self.va_head(h_global_va)
        Va_local = torch.tanh(raw_va) * math.pi
        Va_local = Va_local.view(num_graphs, self.num_kg_base, self.num_boundaries)
        
        # 4. Predict Zk (Discrete Binary Variables via Bernoulli Probabilities)
        h_global_zk = torch.cat([h_global, zk_target, u_zk / 10000.0], dim=1)
        zk_local = self.zk_head(h_global_zk)
        
        # Safely reshape Pg_base using the inferred num_graphs
        num_gens_per_graph = data.gen_mask.sum().item() // num_graphs
        Pg_base = Pg_base.view(num_graphs, num_gens_per_graph)
        
        return Pg_base, Va_local, zk_local

# ==========================================
# 3. UNSUPERVISED ERDŐS ADMM LOSS 
# ==========================================
def compute_zonal_erdos_loss(Pg_base, Va_local, zk_local, Pd_batch, Va_target, u_va, zk_target, u_zk,
                             c2, c1, c0, rho_ADMM=10000.0, lambda_bal=1e5, beta_erdos=50.0):
    
    # 1. Economic Cost
    gen_cost = torch.sum(c2 * (Pg_base ** 2) + c1 * Pg_base + c0, dim=1)
    
    # 2. Physics Power Balance
    total_gen = torch.sum(Pg_base, dim=1)
    total_load = torch.sum(Pd_batch, dim=1)
    balance_penalty = torch.mean((total_gen - total_load) ** 2)
    
    # 3. ADMM Phase Angle Consensus
    va_diff = Va_local - Va_target
    admm_va_linear = torch.sum(u_va * va_diff, dim=[1, 2])
    admm_va_quad = torch.sum((rho_ADMM / 2.0) * (va_diff ** 2), dim=[1, 2])
    
    # 4. ADMM Global Signal Consensus (zk)
    zk_diff = zk_local - zk_target
    admm_zk_linear = torch.sum(u_zk * zk_diff, dim=1)
    admm_zk_quad = torch.sum((rho_ADMM / 2.0) * (zk_diff ** 2), dim=1)
    
    # 5. ERDŐS VARIANCE PENALTY (Unsupervised Binary Forcing)
    erdos_binary_penalty = torch.sum(zk_local * (1.0 - zk_local), dim=1)
    
    # ==============================================================
    # 6. SECURITY PROXY PENALTY (Fix for the "Lazy GNN")
    # Penalizes the network heavily for outputting 0s. 
    # This forces it to activate contingencies to protect the grid!
    # ==============================================================
    security_penalty = torch.sum(1.0 - zk_local, dim=1) 
    
    lambda_admm = 100.0 
    beta_sec = 2000.0  # High penalty weight for ignoring security
    
    admm_loss = torch.mean(admm_va_linear + admm_va_quad + admm_zk_linear + admm_zk_quad)
    
    total_loss = (
        torch.mean(gen_cost) 
        + (lambda_bal * balance_penalty) 
        + (lambda_admm * admm_loss)
        + (beta_erdos * torch.mean(erdos_binary_penalty)) 
        + (beta_sec * torch.mean(security_penalty)) # <-- NEW PENALTY ADDED HERE
    )
    return total_loss

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_agent_gnn(zone_name, z_data, load_data_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM, epochs=150, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing GNN Training for {zone_name.upper()} on {device} ---")
    
    num_kg_base = num_global_kg + 1
    
    c2 = torch.tensor(z_data['gencost']['c2'].values * (baseMVA**2), dtype=torch.float32).to(device)
    c1 = torch.tensor(z_data['gencost']['c1'].values * baseMVA, dtype=torch.float32).to(device)
    c0 = torch.tensor(z_data['gencost']['c0'].values, dtype=torch.float32).to(device)
    
    pyg_dataset = create_pyg_dataset(z_data, load_data_np, baseMVA)
    
    dataloader = DataLoader(pyg_dataset, batch_size=batch_size, shuffle=True)
    
    net = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            current_batch_size = batch.num_graphs
            
            optimizer.zero_grad()
            
            # Simulate ADMM messages from the neighboring zone randomly
            Va_target = (torch.rand(current_batch_size, num_kg_base, num_boundaries).to(device) * 2 * math.pi) - math.pi
            u_va = torch.randn(current_batch_size, num_kg_base, num_boundaries).to(device) * rho_ADMM
            zk_target = torch.rand(current_batch_size, num_global_kg).to(device)
            u_zk = torch.randn(current_batch_size, num_global_kg).to(device) * rho_ADMM
            
            Pg_base, Va_local, zk_local = net(batch, Va_target, u_va, zk_target, u_zk)
            Pd_batch = batch.Pd.view(current_batch_size, -1)
            
            loss = compute_zonal_erdos_loss(Pg_base, Va_local, zk_local, Pd_batch, Va_target, u_va, zk_target, u_zk,
                                            c2, c1, c0, rho_ADMM=rho_ADMM)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Unsupervised Erdős Loss: {epoch_loss/len(dataloader):.2f}")
            
    os.makedirs('data/admm_models', exist_ok=True)
    save_path = f"data/admm_models/{zone_name}_gnn_agent.pth"
    torch.save(net.state_dict(), save_path)
    print(f"[{zone_name.upper()}] GNN Agent Saved to: {save_path}")

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Erdős-GNN Agent")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case14_ieee")
    args = parser.parse_args()

    case_name = args.case
    case_path = f'../excel_outputs/{case_name}.xlsx'
    csv_path = f'data/{case_name}_generated_loads.csv'
    rho_ADMM = 10000.0
    
    print(f"Loading Base Excel Data from {case_path}...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    zero_gen_idx = []
    for num, i in enumerate(case['gen'].Pmax.values / baseMVA):
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0:
            zero_gen_idx.append(num)
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    # -------------------------------------------------------------
    # Dynamically split zones based on Actual Bus IDs 
    # -------------------------------------------------------------
    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses = total_buses[:midpoint]
    zone2_buses = total_buses[midpoint:]
    
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(zonal_data['global_Kg'])
    num_buses_z1 = len(zonal_data['zone1']['bus'])
    
    print(f"Loading Generated Load Profiles from {csv_path}...")
    load_data = pd.read_csv(csv_path).values / baseMVA
    Pd_z1_np = load_data[:, :num_buses_z1]
    Pd_z2_np = load_data[:, num_buses_z1:]
    epochs = 1500
    train_agent_gnn('zone1', zonal_data['zone1'], Pd_z1_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM, epochs=epochs)
    train_agent_gnn('zone2', zonal_data['zone2'], Pd_z2_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM, epochs=epochs)
    
    print("\n--- Both agents successfully trained! ---")