import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import os

# ==========================================
# 1. NETWORK ARCHITECTURE (ZONE 1 AGENT)
# ==========================================
class Zone1_ADMM_Net(nn.Module):
    def __init__(self, num_local_buses, num_local_gens, num_boundaries, num_global_kg, Pmax, Pmin):
        super(Zone1_ADMM_Net, self).__init__()
        self.Pmax = Pmax
        self.Pmin = Pmin
        
        # Kg_and_Base includes the base case (0) + all contingencies (global_Kg)
        self.num_kg_base = num_global_kg + 1 
        self.num_boundaries = num_boundaries
        self.num_global_kg = num_global_kg
        self.num_local_gens = num_local_gens
        
        # INPUTS: Local Load (Pd) + Neighbor's Va (target) + Va Multipliers + Neighbor's zk + zk Multipliers
        # Dimensions: Va needs states for base + all contingencies
        input_dim = num_local_buses + \
                    (self.num_kg_base * num_boundaries) + \
                    (self.num_kg_base * num_boundaries) + \
                    num_global_kg + num_global_kg
                    
        # OUTPUTS: Local Gen (Base) + Proposed Va (Base + Cont) + Proposed zk (Cont)
        self.out_pg_dim = num_local_gens
        self.out_va_dim = self.num_kg_base * num_boundaries
        self.out_zk_dim = num_global_kg
        output_dim = self.out_pg_dim + self.out_va_dim + self.out_zk_dim
        
        hidden_dim = 512 # Increased capacity to handle 55 states simultaneously
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, Pd, Va_target, u_va, zk_target, u_zk):
        # Flatten the 3D target tensors to 2D for the Neural Network
        batch_size = Pd.shape[0]
        Va_target_flat = Va_target.view(batch_size, -1)
        u_va_flat = u_va.view(batch_size, -1)
        
        # Concatenate inputs
        x = torch.cat([Pd, Va_target_flat, u_va_flat, zk_target, u_zk], dim=1)
        raw_out = self.net(x)
        
        # Split outputs
        raw_Pg = raw_out[:, :self.out_pg_dim]
        raw_Va_flat = raw_out[:, self.out_pg_dim : self.out_pg_dim + self.out_va_dim]
        raw_zk = raw_out[:, self.out_pg_dim + self.out_va_dim :]
        
        # Gauge Mapping: Enforce strict physical bounds
        Pg_base = torch.sigmoid(raw_Pg) * (self.Pmax - self.Pmin) + self.Pmin
        Va_local_flat = torch.tanh(raw_Va_flat) * math.pi      # Bounded between -pi and pi
        zk_local = torch.sigmoid(raw_zk)                       # Bounded between 0 and 1
        
        # Reshape Va back to [Batch, Kg_and_Base, Boundary_Buses]
        Va_local = Va_local_flat.view(batch_size, self.num_kg_base, self.num_boundaries)
        
        return Pg_base, Va_local, zk_local

# ==========================================
# 2. THE UNSUPERVISED LOSS FUNCTION
# ==========================================
def compute_zone1_loss(Pg_base, Va_local, zk_local, 
                       Pd, Va_target, u_va, zk_target, u_zk,
                       c2, c1, c0, rho_ADMM=10000.0, 
                       lambda_bal=1e5):
    """
    Direct translation of the 'obj_rule' from the Pyomo CCGA Master script.
    """
    batch_size = Pg_base.shape[0]
    
    # 1. Economic Cost: sum(c2*Pg^2 + c1*Pg + c0)
    gen_cost = torch.sum(c2 * (Pg_base ** 2) + c1 * Pg_base + c0, dim=1)
    
    # 2. Physics Constraints (Simplified Power Balance PINN)
    total_gen = torch.sum(Pg_base, dim=1)
    total_load = torch.sum(Pd, dim=1)
    balance_penalty = torch.mean((total_gen - total_load) ** 2)
    
    # 3. ADMM Augmented Lagrangian for Phase Angles (Va)
    # Equivalent to: sum(u_va * (Va_local - Va_target) + (rho/2)*(Va_local - Va_target)^2)
    va_diff = Va_local - Va_target
    admm_va_linear = torch.sum(u_va * va_diff, dim=[1, 2])
    admm_va_quad = torch.sum((rho_ADMM / 2.0) * (va_diff ** 2), dim=[1, 2])
    
    # 4. ADMM Augmented Lagrangian for Global Signal (zk)
    # Equivalent to: sum(u_zk * (zk - neighbor_zk) + (rho/2)*(zk - neighbor_zk)^2)
    zk_diff = zk_local - zk_target
    admm_zk_linear = torch.sum(u_zk * zk_diff, dim=1)
    admm_zk_quad = torch.sum((rho_ADMM / 2.0) * (zk_diff ** 2), dim=1)
    
    # Total Objective matching your MILP
    admm_loss = torch.mean(admm_va_linear + admm_va_quad + admm_zk_linear + admm_zk_quad)
    total_loss = torch.mean(gen_cost) + (lambda_bal * balance_penalty) + admm_loss
    
    return total_loss

# ==========================================
# 3. TRAINING LOOP (DOMAIN RANDOMIZATION)
# ==========================================
def train_zone1_unsupervised():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. Configuration based on case118 splits ---
    case_name = 'case118'
    baseMVA = 100.0
    rho_ADMM = 10000.0
    
    # Assuming standard splits from your script
    num_local_buses = 59     # Zone 1 buses (1 to 59)
    num_local_gens = 27      # Approx local gens
    num_boundaries = 9       # Number of tie-line buses
    num_global_kg = 54       # Total generators in the system
    num_kg_base = num_global_kg + 1
    
    # Generate mock tensors (In production, load these dynamically from zonal_data['zone1']['gen'])
    Pmax = torch.ones(num_local_gens, dtype=torch.float32).to(device) * (500.0 / baseMVA)
    Pmin = torch.zeros(num_local_gens, dtype=torch.float32).to(device)
    c2 = torch.rand(num_local_gens).to(device) * (0.05 * baseMVA**2)
    c1 = torch.rand(num_local_gens).to(device) * (20.0 * baseMVA)
    c0 = torch.rand(num_local_gens).to(device) * 100.0
    
    # --- B. Initialize Model ---
    zone1_net = Zone1_ADMM_Net(num_local_buses, num_local_gens, num_boundaries, num_global_kg, Pmax, Pmin).to(device)
    optimizer = optim.Adam(zone1_net.parameters(), lr=1e-3)
    
    # --- C. Load Generated Load Profiles (No y_final labels needed!) ---
    print("Loading Load Scenarios...")
    csv_path = 'data/118_ieee_generated_loads.csv'
    
    # Fallback to random data if CSV is not found for script testing
    if os.path.exists(csv_path):
        load_data = pd.read_csv(csv_path).values / baseMVA
        zone1_loads = load_data[:, :num_local_buses]
    else:
        print(f"Warning: {csv_path} not found. Using simulated load data.")
        zone1_loads = np.random.rand(1000, num_local_buses) * (150.0 / baseMVA)
        
    dataset = torch.utils.data.TensorDataset(torch.tensor(zone1_loads, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # --- D. Unsupervised Training Loop ---
    print("Starting Unsupervised ADMM Training for Zone 1...")
    epochs = 150
    
    zone1_net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_Pd in dataloader:
            batch_Pd = batch_Pd[0].to(device)
            batch_size = batch_Pd.shape[0]
            
            optimizer.zero_grad()
            
            # DOMAIN RANDOMIZATION: Simulate Zone 2's ADMM boundary messages
            # Target Phase Angles (Va_target) across all states
            Va_target = (torch.rand(batch_size, num_kg_base, num_boundaries).to(device) * 2 * math.pi) - math.pi
            u_va = torch.randn(batch_size, num_kg_base, num_boundaries).to(device) * 10.0
            
            # Target Global Signals (zk_target) across all contingencies
            zk_target = torch.rand(batch_size, num_global_kg).to(device)
            u_zk = torch.randn(batch_size, num_global_kg).to(device) * 10.0
            
            # Forward Pass
            Pg_base, Va_local, zk_local = zone1_net(batch_Pd, Va_target, u_va, zk_target, u_zk)
            
            # Compute Unsupervised ADMM Loss
            loss = compute_zone1_loss(Pg_base, Va_local, zk_local, 
                                      batch_Pd, Va_target, u_va, zk_target, u_zk,
                                      c2, c1, c0, rho_ADMM=rho_ADMM)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Unsupervised Loss: {epoch_loss/len(dataloader):.2f}")
            
    # Save the trained Zone 1 Agent
    os.makedirs('data/admm_models', exist_ok=True)
    torch.save(zone1_net.state_dict(), "data/admm_models/zone1_agent.pth")
    print("Zone 1 Neural Network Saved Successfully!")

if __name__ == "__main__":
    train_zone1_unsupervised()