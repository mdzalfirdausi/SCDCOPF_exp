import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import os
import time

# ==========================================
# 1. DATA PREPARATION (From your original scripts)
# ==========================================
def create_zonal_data(case, zone1_buses, zone2_buses):
    """Splits the grid into zones (Extracted from run_admm_zonal.py)"""
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
            
    zonal_data['boundary_buses'] = list(boundary_buses)
            
    zonal_data['zone1']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['branch'] = branch_df[(branch_df['bus_i'].isin(zone1_buses)) & (branch_df['bus_j'].isin(zone1_buses))].copy()
    
    zonal_data['zone2']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['branch'] = branch_df[(branch_df['bus_i'].isin(zone2_buses)) & (branch_df['bus_j'].isin(zone2_buses))].copy()
    
    for z in ['zone1', 'zone2']:
        zonal_data[z]['baseMVA'] = case['baseMVA']
        zonal_data[z]['global_Kg'] = zonal_data['global_Kg']
    
    return zonal_data

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class Zone_ADMM_Net(nn.Module):
    def __init__(self, num_local_buses, num_local_gens, num_boundaries, num_global_kg, Pmax, Pmin):
        super(Zone_ADMM_Net, self).__init__()
        
        # FIX: Registering these as buffers ensures they automatically move to the GPU
        # when you call net_z1.to(device) in the main block.
        self.register_buffer('Pmax', torch.tensor(Pmax, dtype=torch.float32))
        self.register_buffer('Pmin', torch.tensor(Pmin, dtype=torch.float32))
        
        self.num_kg_base = num_global_kg + 1 
        self.num_boundaries = num_boundaries
        self.out_zk_dim = num_global_kg
        
        input_dim = num_local_buses + (self.num_kg_base * num_boundaries) * 2 + num_global_kg * 2
        self.out_pg_dim = num_local_gens
        self.out_va_dim = self.num_kg_base * num_boundaries
        output_dim = self.out_pg_dim + self.out_va_dim + self.out_zk_dim
        
        hidden_dim = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, Pd, Va_target, u_va, zk_target, u_zk):
        batch_size = Pd.shape[0]
        x = torch.cat([Pd, Va_target.view(batch_size, -1), u_va.view(batch_size, -1), zk_target, u_zk], dim=1)
        raw_out = self.net(x)
        
        raw_Pg = raw_out[:, :self.out_pg_dim]
        raw_Va_flat = raw_out[:, self.out_pg_dim : self.out_pg_dim + self.out_va_dim]
        raw_zk = raw_out[:, self.out_pg_dim + self.out_va_dim :]
        
        # Now Pmax and Pmin are guaranteed to be on the exact same device as raw_Pg
        Pg_base = torch.sigmoid(raw_Pg) * (self.Pmax - self.Pmin) + self.Pmin
        Va_local = (torch.tanh(raw_Va_flat) * math.pi).view(batch_size, self.num_kg_base, self.num_boundaries)
        zk_local = torch.sigmoid(raw_zk)                       
        
        return Pg_base, Va_local, zk_local
# ==========================================
# 3. NEURAL ADMM DEPLOYMENT LOOP
# ==========================================
def run_neural_distributed_admm(Pd_z1, Pd_z2, net_z1, net_z2, rho=10000.0, max_iters=100, tol=5e-3):
    device = Pd_z1.device
    batch_size = Pd_z1.shape[0]
    
    num_boundaries = net_z1.num_boundaries
    num_kg_base = net_z1.num_kg_base
    num_global_kg = net_z1.out_zk_dim
    
    Va_z1 = torch.zeros(batch_size, num_kg_base, num_boundaries, device=device)
    Va_z2 = torch.zeros(batch_size, num_kg_base, num_boundaries, device=device)
    u_va  = torch.zeros(batch_size, num_kg_base, num_boundaries, device=device)
    
    zk_z1 = torch.zeros(batch_size, num_global_kg, device=device)
    zk_z2 = torch.zeros(batch_size, num_global_kg, device=device)
    u_zk  = torch.zeros(batch_size, num_global_kg, device=device)
    
    print("\nStarting Neural ADMM Loop...")
    start_time = time.time()
    
    with torch.no_grad():
        for itr in range(1, max_iters + 1):
            Pg_z1, Va_z1, zk_z1 = net_z1(Pd_z1, Va_z2, u_va, zk_z2, u_zk)
            Pg_z2, Va_z2, zk_z2 = net_z2(Pd_z2, Va_z1, -u_va, zk_z1, -u_zk)
            
            res_va = torch.sum((Va_z1 - Va_z2)**2, dim=[1, 2])
            res_zk = torch.sum((zk_z1 - zk_z2)**2, dim=1)
            primal_residual = torch.sqrt(res_va + res_zk).mean().item()
            
            if itr % 10 == 0 or itr == 1:
                print(f"--- Iteration {itr:3d} --- Primal Residual: {primal_residual:.6f}")
                
            if primal_residual <= tol:
                print(f"\nNeural ADMM Converged! ({primal_residual:.6f} <= {tol})")
                break
                
            u_va = u_va + rho * (Va_z1 - Va_z2)
            u_zk = u_zk + rho * (zk_z1 - zk_z2)
            
    print(f"D-SCDCOPF Complete in {time.time() - start_time:.4f} seconds!")
    return Pg_z1, Va_z1, zk_z1, Pg_z2, Va_z2, zk_z2

# ==========================================
# 4. EXCEL EXPORT FUNCTION
# ==========================================
def save_neural_results_to_excel(Pg_z1, zk_z1, Pg_z2, zonal_data, case_name, batch_idx=0, base_dir="data"):
    output_dir = os.path.join(base_dir, "admm_result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{case_name}_neural_admm_results.xlsx")
    
    print(f"\nExtracting Neural Tensors to {output_path}...")

    pg_z1_np = Pg_z1[batch_idx].cpu().numpy()
    pg_z2_np = Pg_z2[batch_idx].cpu().numpy()
    zk_np = zk_z1[batch_idx].cpu().numpy()
    
    gen_data = []
    for idx, gen_id in enumerate(zonal_data['zone1']['gen']['gen_ID'].tolist()):
        gen_data.append({'Zone': 'Zone_1', 'Gen_ID': gen_id, 'Pg_base': pg_z1_np[idx]})
    for idx, gen_id in enumerate(zonal_data['zone2']['gen']['gen_ID'].tolist()):
        gen_data.append({'Zone': 'Zone_2', 'Gen_ID': gen_id, 'Pg_base': pg_z2_np[idx]})
        
    df_gen = pd.DataFrame(gen_data).sort_values('Gen_ID')

    zk_data = [{'Contingency_k': k, 'zk_signal': zk_np[idx]} for idx, k in enumerate(zonal_data['global_Kg'])]
    df_zk = pd.DataFrame(zk_data)

    with pd.ExcelWriter(output_path) as writer:
        df_gen.to_excel(writer, sheet_name='Generators', index=False)
        df_zk.to_excel(writer, sheet_name='Global_Signal', index=False)
        
    print(f"Success: Neural Results saved!")

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_name = 'pglib_opf_case118_ieee'
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    # 1. Load Excel Data
    print(f"Loading data from {case_path}...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    # Clean zero-generation limits
    zero_gen_idx = []
    for num, i in enumerate(case['gen'].Pmax.values / baseMVA):
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0:
            zero_gen_idx.append(num)
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    
    # Define Zones
    zone1_buses = list(range(1, 60))    
    zone2_buses = list(range(60, 119))  
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    
    # 2. Extract Dimensions for the Neural Networks
    num_buses_z1 = len(zonal_data['zone1']['bus'])
    num_gens_z1 = len(zonal_data['zone1']['gen'])
    Pmax_z1 = zonal_data['zone1']['gen']['Pmax'].values / baseMVA
    Pmin_z1 = zonal_data['zone1']['gen']['Pmin'].values / baseMVA
    
    num_buses_z2 = len(zonal_data['zone2']['bus'])
    num_gens_z2 = len(zonal_data['zone2']['gen'])
    Pmax_z2 = zonal_data['zone2']['gen']['Pmax'].values / baseMVA
    Pmin_z2 = zonal_data['zone2']['gen']['Pmin'].values / baseMVA
    
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(zonal_data['global_Kg'])
    
    # 3. Initialize Networks and Load Weights
    print("Initializing PyTorch Models...")
    net_z1 = Zone_ADMM_Net(num_buses_z1, num_gens_z1, num_boundaries, num_global_kg, Pmax_z1, Pmin_z1).to(device)
    net_z2 = Zone_ADMM_Net(num_buses_z2, num_gens_z2, num_boundaries, num_global_kg, Pmax_z2, Pmin_z2).to(device)
    
    try:
        # strict=False tells PyTorch to ignore the missing Pmax and Pmin keys in the old save files
        net_z1.load_state_dict(torch.load("data/admm_models/zone1_agent.pth", map_location=device), strict=False)
        net_z2.load_state_dict(torch.load("data/admm_models/zone2_agent.pth", map_location=device), strict=False)
        print("Trained weights loaded successfully.")
    except FileNotFoundError:
        print("\nWARNING: Trained weights not found. Using untrained (random) networks for demonstration.")
    
    net_z1.eval()
    net_z2.eval()
    
    # 4. Load Real-Time Load Scenarios
    print("Injecting Load Scenario...")
    csv_path = 'data/118_ieee_generated_loads.csv'
    if os.path.exists(csv_path):
        load_data = pd.read_csv(csv_path).values / baseMVA
        # Grab the first scenario (row 0)
        Pd_z1_np = load_data[0:1, :num_buses_z1]
        Pd_z2_np = load_data[0:1, num_buses_z1:]
    else:
        print("CSV not found. Using mock load data.")
        Pd_z1_np = np.random.rand(1, num_buses_z1) * 1.5
        Pd_z2_np = np.random.rand(1, num_buses_z2) * 1.5
        
    Pd_z1 = torch.tensor(Pd_z1_np, dtype=torch.float32).to(device)
    Pd_z2 = torch.tensor(Pd_z2_np, dtype=torch.float32).to(device)
    
    # 5. Run the Distributed Neural Negotiation
    Pg_z1, Va_z1, zk_z1, Pg_z2, Va_z2, zk_z2 = run_neural_distributed_admm(
        Pd_z1, Pd_z2, net_z1, net_z2, rho=10000.0
    )
    
    # 6. Export to Excel
    save_neural_results_to_excel(
        Pg_z1, zk_z1, Pg_z2, zonal_data, case_name, batch_idx=0
    )