import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import os

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def create_zonal_data(case, zone1_buses, zone2_buses):
    """Splits the grid into zones (matches your original Pyomo ADMM logic)."""
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

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class Zone_ADMM_Net(nn.Module):
    def __init__(self, num_local_buses, num_local_gens, num_boundaries, num_global_kg, Pmax, Pmin):
        super(Zone_ADMM_Net, self).__init__()
        self.Pmax = torch.tensor(Pmax, dtype=torch.float32)
        self.Pmin = torch.tensor(Pmin, dtype=torch.float32)
        
        # Kg_and_Base includes the base case (0) + all contingencies
        self.num_kg_base = num_global_kg + 1 
        self.num_boundaries = num_boundaries
        self.out_zk_dim = num_global_kg
        
        # Inputs: Load + Va_target + u_va + zk_target + u_zk
        input_dim = num_local_buses + (self.num_kg_base * num_boundaries) * 2 + num_global_kg * 2
        
        self.out_pg_dim = num_local_gens
        self.out_va_dim = self.num_kg_base * num_boundaries
        output_dim = self.out_pg_dim + self.out_va_dim + self.out_zk_dim
        
        hidden_dim = 2048
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, Pd, Va_target, u_va, zk_target, u_zk):
        batch_size = Pd.shape[0]
        
        # NORMALIZE MULTIPLIERS: Divide by rho_ADMM (10000.0) so the NN isn't blinded by massive numbers
        u_va_norm = u_va.view(batch_size, -1) / 10000.0
        u_zk_norm = u_zk / 10000.0
        
        x = torch.cat([Pd, Va_target.view(batch_size, -1), u_va_norm, zk_target, u_zk_norm], dim=1)
        raw_out = self.net(x)
        
        raw_Pg = raw_out[:, :self.out_pg_dim]
        raw_Va_flat = raw_out[:, self.out_pg_dim : self.out_pg_dim + self.out_va_dim]
        raw_zk = raw_out[:, self.out_pg_dim + self.out_va_dim :]
        
        Pg_base = torch.sigmoid(raw_Pg) * (self.Pmax.to(x.device) - self.Pmin.to(x.device)) + self.Pmin.to(x.device)
        Va_local = (torch.tanh(raw_Va_flat) * math.pi).view(batch_size, self.num_kg_base, self.num_boundaries)
        zk_local = torch.sigmoid(raw_zk)                       
        
        return Pg_base, Va_local, zk_local

# ==========================================
# 3. UNSUPERVISED ADMM LOSS FUNCTION
# ==========================================
def compute_zonal_loss(Pg_base, Va_local, zk_local, Pd, Va_target, u_va, zk_target, u_zk,
                       c2, c1, c0, rho_ADMM=10000.0, lambda_bal=1e5):
    """Calculates the loss by mapping Pyomo obj_rule and constraints to PyTorch."""
    
    # 1. Economic Cost: sum(c2*Pg^2 + c1*Pg + c0)
    gen_cost = torch.sum(c2 * (Pg_base ** 2) + c1 * Pg_base + c0, dim=1)
    
    # 2. Physics Constraints (Simplified Power Balance)
    total_gen = torch.sum(Pg_base, dim=1)
    total_load = torch.sum(Pd, dim=1)
    balance_penalty = torch.mean((total_gen - total_load) ** 2)
    
    # 3. ADMM Phase Angle Consensus (Va)
    va_diff = Va_local - Va_target
    admm_va_linear = torch.sum(u_va * va_diff, dim=[1, 2])
    admm_va_quad = torch.sum((rho_ADMM / 2.0) * (va_diff ** 2), dim=[1, 2])
    
    # 4. ADMM Global Signal Consensus (zk)
    zk_diff = zk_local - zk_target
    admm_zk_linear = torch.sum(u_zk * zk_diff, dim=1)
    admm_zk_quad = torch.sum((rho_ADMM / 2.0) * (zk_diff ** 2), dim=1)
    
    # FIX: Apply a massive penalty multiplier (100.0) so the NN respects the boundary!
    lambda_admm = 100.0 
    
    admm_loss = torch.mean(admm_va_linear + admm_va_quad + admm_zk_linear + admm_zk_quad)
    
    total_loss = torch.mean(gen_cost) + (lambda_bal * balance_penalty) + (lambda_admm * admm_loss)
    
    return total_loss

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_agent(zone_name, z_data, load_data_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM, epochs=150, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing Training for {zone_name.upper()} on {device} ---")
    
    # Extract grid parameters
    num_local_buses = len(z_data['bus'])
    num_local_gens = len(z_data['gen'])
    num_kg_base = num_global_kg + 1
    
    Pmax = z_data['gen']['Pmax'].values / baseMVA
    Pmin = z_data['gen']['Pmin'].values / baseMVA
    
    c2_np = z_data['gencost']['c2'].values * (baseMVA**2)
    c1_np = z_data['gencost']['c1'].values * baseMVA
    c0_np = z_data['gencost']['c0'].values
    
    c2 = torch.tensor(c2_np, dtype=torch.float32).to(device)
    c1 = torch.tensor(c1_np, dtype=torch.float32).to(device)
    c0 = torch.tensor(c0_np, dtype=torch.float32).to(device)
    
    # Setup Data Loader
    dataset = torch.utils.data.TensorDataset(torch.tensor(load_data_np, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    net = Zone_ADMM_Net(num_local_buses, num_local_gens, num_boundaries, num_global_kg, Pmax, Pmin).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # Domain Randomization Training Loop
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_Pd in dataloader:
            batch_Pd = batch_Pd[0].to(device)
            current_batch_size = batch_Pd.shape[0]
            
            optimizer.zero_grad()
            
            # Simulate ADMM messages from the neighboring zone randomly
            Va_target = (torch.rand(current_batch_size, num_kg_base, num_boundaries).to(device) * 2 * math.pi) - math.pi
            
            # FIX: Train the network to expect massive ADMM multipliers!
            u_va = torch.randn(current_batch_size, num_kg_base, num_boundaries).to(device) * rho_ADMM
            
            zk_target = torch.rand(current_batch_size, num_global_kg).to(device)
            u_zk = torch.randn(current_batch_size, num_global_kg).to(device) * rho_ADMM
            
            # Forward Pass
            Pg_base, Va_local, zk_local = net(batch_Pd, Va_target, u_va, zk_target, u_zk)
            
            # Compute Loss
            loss = compute_zonal_loss(Pg_base, Va_local, zk_local, batch_Pd, Va_target, u_va, zk_target, u_zk,
                                      c2, c1, c0, rho_ADMM=rho_ADMM)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Unsupervised Loss: {epoch_loss/len(dataloader):.2f}")
            
    # Save Model
    os.makedirs('data/admm_models', exist_ok=True)
    save_path = f"data/admm_models/{zone_name}_agent.pth"
    torch.save(net.state_dict(), save_path)
    print(f"[{zone_name.upper()}] Agent Saved to: {save_path}")

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    case_name = 'pglib_opf_case118_ieee'
    case_path = f'../excel_outputs/{case_name}.xlsx'
    csv_path = 'data/pglib_opf_case118_ieee_generated_loads.csv'
    rho_ADMM = 10000.0
    
    print(f"Loading Base Excel Data from {case_path}...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    # Clean zero-generation limits
    zero_gen_idx = []
    for num, i in enumerate(case['gen'].Pmax.values / baseMVA):
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0:
            zero_gen_idx.append(num)
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    # Extract Zonal Topology
    zone1_buses = list(range(1, 60))    
    zone2_buses = list(range(60, 119))  
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(zonal_data['global_Kg'])
    num_buses_z1 = len(zonal_data['zone1']['bus'])
    
    print(f"Loading Generated Load Profiles from {csv_path}...")
    if os.path.exists(csv_path):
        load_data = pd.read_csv(csv_path).values / baseMVA
        Pd_z1_np = load_data[:, :num_buses_z1]
        Pd_z2_np = load_data[:, num_buses_z1:]
    else:
        print(f"WARNING: {csv_path} not found. Generating dummy load scenarios.")
        load_data = np.random.rand(1000, 118) * (150.0 / baseMVA)
        Pd_z1_np = load_data[:, :num_buses_z1]
        Pd_z2_np = load_data[:, num_buses_z1:]
    
    # Train Zone 1
    train_agent('zone1', zonal_data['zone1'], Pd_z1_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM)
    
    # Train Zone 2
    train_agent('zone2', zonal_data['zone2'], Pd_z2_np, num_boundaries, num_global_kg, baseMVA, rho_ADMM)
    
    print("\n--- Both agents successfully trained! ---")
    print("You can now run 'deploy_unsupervised.py' to execute the fast neural ADMM.")