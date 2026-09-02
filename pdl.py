import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from dcopf_model import build_ptdf, build_lodf

# ==========================================
# 1. DIFFERENTIABLE PHYSICS LAYERS[cite: 6]
# ==========================================

def power_balance_repair_layer(g_raw, d_total, pmin, pmax):
    """Ensures the base-case dispatch perfectly meets demand."""
    g_total = g_raw.sum(dim=1, keepdim=True)
    pmax_total = pmax.sum(dim=1, keepdim=True)
    pmin_total = pmin.sum(dim=1, keepdim=True)

    zeta_up = (d_total - g_total) / (pmax_total - g_total + 1e-9)
    zeta_down = (g_total - d_total) / (g_total - pmin_total + 1e-9)

    condition = g_total < d_total
    g_repaired = torch.where(
        condition,
        (1 - zeta_up) * g_raw + zeta_up * pmax,
        (1 - zeta_down) * g_raw + zeta_down * pmin
    )
    return g_repaired

def compute_physics_loss(g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map):
    """
    Computes ALM Constraint Mismatches (h_x) and Soft Slack Violations.
    """
    batch_size, num_gens = g_star.shape
    M_eta = 1500.0 # Heavy penalty for soft thermal limits
    d_total = d_bus.sum(dim=1, keepdim=True)
    g_bus = torch.matmul(g_star, bus_gen_map.T) 
    
    # 1. BASE CASE PHYSICS
    net_injections = g_bus - d_bus
    f_star = torch.matmul(net_injections, PTDF.T)
    eta_0 = F.relu(torch.abs(f_star) - f_max)
    
    # 2. LINE CONTINGENCY PHYSICS (Ke)
    f_star_outaged = f_star.unsqueeze(1) 
    f_k_e = f_star.unsqueeze(2) + LODF.unsqueeze(0) * f_star_outaged 
    
    # FIX: Use .view(1, -1, 1) to broadcast the 1D tensor to [1, E, 1]
    eta_k_e = F.relu(torch.abs(f_k_e) - f_max.view(1, -1, 1))
    
    # 3. GENERATOR CONTINGENCY PHYSICS (Kg) & ALM MISMATCH
    # Binary Search Layer logic (APR limits)
    n_low = torch.zeros(batch_size, num_gens, 1, device=g_star.device)
    n_high = torch.ones(batch_size, num_gens, 1, device=g_star.device)
    n_k = torch.full((batch_size, num_gens, 1), 0.5, device=g_star.device)
    droop_slope = gamma * pmax 
    
    for _ in range(20):
        g_prov = g_star.unsqueeze(1) + n_k * droop_slope.unsqueeze(1)
        
        # FIX: Clamp the floor to 0.0, then take the element-wise minimum with Pmax
        g_k = torch.min(g_prov.clamp(min=0.0), pmax.unsqueeze(1))
        
        mask = torch.eye(num_gens, device=g_star.device).bool().unsqueeze(0)
        g_k = g_k.masked_fill(mask, 0.0)
        
        current_generation = g_k.sum(dim=2, keepdim=True)
        mismatch = current_generation - d_total.unsqueeze(1)
        
        n_low = torch.where(mismatch < 0, n_k, n_low)
        n_high = torch.where(mismatch > 0, n_k, n_high)
        n_k = (n_low + n_high) / 2.0

    # Final forward pass for g_k
    g_prov = g_star.unsqueeze(1) + n_k * droop_slope.unsqueeze(1)
    
    # FIX: Apply the same separated bounds logic here
    g_k = torch.min(g_prov.clamp(min=0.0), pmax.unsqueeze(1))
    
    g_k = g_k.masked_fill(mask, 0.0)
    
    # h_x(y): Exact Power Balance Mismatch under Generator Contingencies[cite: 6]
    h_x = (g_k.sum(dim=2) - d_total) # Shape: (Batch, Kg)
    
    g_k_bus = torch.matmul(g_k, bus_gen_map.T) 
    net_injections_k = g_k_bus - d_bus.unsqueeze(1) 
    f_k_g = torch.matmul(net_injections_k, PTDF.T) 
    
    # FIX: Just use f_max directly. PyTorch naturally broadcasts [E] to [B, Kg, E]
    eta_k_g = F.relu(torch.abs(f_k_g) - f_max)
    
    gen_cost = torch.sum(c1 * g_star + c0, dim=1)
    total_slack = eta_0.sum(dim=1) + eta_k_e.sum(dim=(1,2)) + eta_k_g.sum(dim=(1,2))
    
    return gen_cost, total_slack, h_x


# ==========================================
# 2. PRIMAL AND DUAL GRAPH NEURAL NETWORKS
# ==========================================

class PDLPrimalNet(nn.Module):
    """Predicts the continuous base-case active power dispatch."""
    def __init__(self, num_node_features, num_gens):
        super(PDLPrimalNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_gens)
        self.layer_norm = nn.LayerNorm(128) # Added per 2025 PDL architecture guidelines[cite: 6]

    def forward(self, x, edge_index, batch, pmin, pmax):
        h = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        h = F.leaky_relu(self.conv2(h, edge_index), 0.1)
        h_graph = self.layer_norm(global_mean_pool(h, batch))
        
        h_fc = F.leaky_relu(self.fc1(h_graph), 0.1)
        out_scaled = torch.sigmoid(self.fc2(h_fc))
        return pmin + out_scaled * (pmax - pmin)

class PDLDualNet(nn.Module):
    """Predicts the Lagrangian Multipliers (lambda) for generator contingencies."""
    def __init__(self, num_node_features, num_kg):
        super(PDLDualNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_kg) # Outputs 1 multiplier per generator contingency

    def forward(self, x, edge_index, batch):
        h = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        h = F.leaky_relu(self.conv2(h, edge_index), 0.1)
        h_graph = global_mean_pool(h, batch)
        
        h_fc = F.leaky_relu(self.fc1(h_graph), 0.1)
        return self.fc2(h_fc) # Lambda can be positive or negative


# ==========================================
# 3. DATASET PREPARATION
# ==========================================
def create_pdl_dataset(case, load_data_np, baseMVA):
    bus_df = case['bus']
    branch_df = case['branch']
    num_buses = len(bus_df)
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_df['bus_i'].values)}
    
    edge_source = [bus_idx_map[i] for i in branch_df['bus_i'].values]
    edge_target = [bus_idx_map[j] for j in branch_df['bus_j'].values]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)
    
    dataset = []
    for s in range(len(load_data_np)):
        Pd = torch.tensor(load_data_np[s], dtype=torch.float32).view(-1, 1)
        data = Data(x=Pd, edge_index=edge_index)
        data.Pd = Pd.view(-1)
        dataset.append(data)
    return dataset


# ==========================================
# 4. PRIMAL-DUAL TRAINING ENGINE
# ==========================================
def train_pdl_scopf(case_name, outer_K=20, inner_L=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing Primal-Dual ALM Training for {case_name.upper()} on {device} ---")
    
    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) 
                    if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or 
                    (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    bus_list = sorted(case['bus']['bus_i'].tolist())
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    
    PTDF_np, _ = build_ptdf(case['bus'], case['branch'], ref_bus)
    LODF_np = build_lodf(PTDF_np, case['branch'], bus_list)
    
    PTDF = torch.tensor(PTDF_np, dtype=torch.float32, device=device)
    LODF = torch.tensor(LODF_np, dtype=torch.float32, device=device)
    f_max = torch.tensor(case['branch']['rateA'].values / baseMVA, dtype=torch.float32, device=device)
    
    num_gens = len(case['gen'])
    pmax = torch.tensor(case['gen']['Pmax'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    pmin = torch.tensor(case['gen']['Pmin'].values / baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    gamma = torch.tensor([1.0] * num_gens, dtype=torch.float32, device=device).unsqueeze(0)
    c1 = torch.tensor(case['gencost']['c1'].values * baseMVA, dtype=torch.float32, device=device).unsqueeze(0)
    c0 = torch.tensor(case['gencost']['c0'].values, dtype=torch.float32, device=device).unsqueeze(0)
    
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(bus_list)}
    bus_gen_map_np = np.zeros((len(bus_list), num_gens))
    for j, bus_i in enumerate(case['gen']['bus_i']):
        bus_gen_map_np[bus_idx_map[bus_i], j] = 1.0
    bus_gen_map = torch.tensor(bus_gen_map_np, dtype=torch.float32, device=device)
    
    csv_path = f"data/{case_name}_generated_data.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/{case_name}_generated_loads.csv"
        
    # Read the CSV as a DataFrame first
    df_csv = pd.read_csv(csv_path)
    
    # Extract ONLY the active load (Pd) columns in the exact order of the bus list
    pd_cols = [f"Bus_{b}_Pd" for b in bus_list]
    if all(col in df_csv.columns for col in pd_cols):
        load_data_np = df_csv[pd_cols].values / baseMVA
    else:
        # Fallback if using older CSV formats: grab just the first N columns
        load_data_np = df_csv.iloc[:, :len(bus_list)].values / baseMVA
        
    dataset = create_pdl_dataset(case, load_data_np, baseMVA)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Primal and Dual Networks
    net_P = PDLPrimalNet(num_node_features=1, num_gens=num_gens).to(device)
    net_D = PDLDualNet(num_node_features=1, num_kg=num_gens).to(device)
    
    opt_P = optim.Adam(net_P.parameters(), lr=1e-4)
    opt_D = optim.Adam(net_D.parameters(), lr=1e-4)
    
    # Algorithm 2 Parameters[cite: 6]
    rho = 0.1 
    rho_max = 1e8
    tau = 0.9
    alpha = 2.0
    v_prev = float('inf')
    
    print("\nStarting PDL Training (Augmented Lagrangian Method)...")
    for k in range(outer_K):
        
        # --- 1. PRIMAL LEARNING PHASE ---
        net_P.train()
        net_D.eval()
        for _ in range(inner_L):
            for batch in dataloader:
                batch = batch.to(device)
                opt_P.zero_grad()
                
                g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
                d_bus = batch.Pd.view(batch.num_graphs, -1)
                g_star = power_balance_repair_layer(g_raw, d_bus.sum(dim=1, keepdim=True), pmin, pmax)
                
                gen_cost, total_slack, h_x = compute_physics_loss(
                    g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map
                )
                
                # Get Lagrange Multipliers (Detached so gradients don't flow to Dual Net)
                lambdas = net_D(batch.x, batch.edge_index, batch.batch).detach()
                
                # ALM Primal Loss[cite: 6]
                alm_linear = torch.sum(lambdas * h_x, dim=1)
                alm_quad = torch.sum((rho / 2.0) * (h_x ** 2), dim=1)
                
                loss_P = torch.mean((gen_cost + 1500.0 * total_slack) / 1e5 + alm_linear + alm_quad)
                loss_P.backward()
                opt_P.step()
                
        # --- 2. DUAL LEARNING PHASE ---
        net_P.eval()
        net_D.train()
        net_D_frozen = copy.deepcopy(net_D) # Fixed network for Lagrangian tracking[cite: 6]
        
        max_h_x_violation = 0.0 # Track for rho update
        
        for _ in range(inner_L):
            for batch in dataloader:
                batch = batch.to(device)
                opt_D.zero_grad()
                
                with torch.no_grad():
                    g_raw = net_P(batch.x, batch.edge_index, batch.batch, pmin, pmax)
                    d_bus = batch.Pd.view(batch.num_graphs, -1)
                    g_star = power_balance_repair_layer(g_raw, d_bus.sum(dim=1, keepdim=True), pmin, pmax)
                    _, _, h_x = compute_physics_loss(g_star, d_bus, c1, c0, PTDF, LODF, f_max, pmax, gamma, bus_gen_map)
                    
                    lambda_k = net_D_frozen(batch.x, batch.edge_index, batch.batch)
                    
                    max_h_x_violation = max(max_h_x_violation, torch.max(torch.abs(h_x)).item())

                # ALM Dual Update Target[cite: 6]
                lambda_est = net_D(batch.x, batch.edge_index, batch.batch)
                
                # The dual penalty coefficient is fixed at 1e-1[cite: 6]
                target = (lambda_k + 0.1 * h_x).detach()
                loss_D = F.mse_loss(lambda_est, target)
                
                loss_D.backward()
                opt_D.step()

        # --- 3. DYNAMIC PENALTY UPDATE ---
        v_k = max_h_x_violation
        if v_k > tau * v_prev:
            rho = min(alpha * rho, rho_max)
        v_prev = v_k
        
        print(f"Outer Iteration {k+1:2d}/{outer_K} | rho: {rho:.2e} | Max Gen-Contingency Mismatch: {v_k:.4f} p.u.")
            
    os.makedirs('data/models', exist_ok=True)
    torch.save(net_P.state_dict(), f"data/models/{case_name}_pdl_primal.pth")
    torch.save(net_D.state_dict(), f"data/models/{case_name}_pdl_dual.pth")
    print("\n[+] PDL Primal and Dual Networks successfully trained and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Self-Supervised PDL-SCOPF")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case300_ieee")
    args = parser.parse_args()
    
    # Outer_K = 20, Inner_L = 2000 (Adjusted default for practical dataset scaling)
    train_pdl_scopf(args.case, outer_K=20, inner_L=5, batch_size=32)