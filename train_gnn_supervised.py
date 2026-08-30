import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Import your GNN architectures and data builders
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data, create_pyg_dataset

def train_agent(agent_name, model, train_loader, val_loader, device, num_global_kg, num_boundaries, epochs=50, lr=1e-3):
    """Standard Supervised Training Loop with BCE Loss"""
    print(f"\n=======================================================")
    print(f" TRAINING: {agent_name.upper()}")
    print(f"=======================================================")
    
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Dummy tensors for ADMM communication variables (not used in one-shot inference)
            va_dummy = torch.zeros(batch.num_graphs, num_global_kg + 1, num_boundaries, device=device)
            zk_dummy = torch.zeros(batch.num_graphs, num_global_kg, device=device)
            
            _, _, zk_prob = model(batch, va_dummy, va_dummy, zk_dummy, zk_dummy)
            
            # zk_prob shape: (batch, num_global_kg, 1) -> squeeze(-1) makes it (batch, num_global_kg)
            # batch.y shape: (batch, num_global_kg)
            loss = criterion(zk_prob.squeeze(-1), batch.y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct, total_elements = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                va_dummy = torch.zeros(batch.num_graphs, num_global_kg + 1, num_boundaries, device=device)
                zk_dummy = torch.zeros(batch.num_graphs, num_global_kg, device=device)
                
                _, _, zk_prob = model(batch, va_dummy, va_dummy, zk_dummy, zk_dummy)
                
                loss = criterion(zk_prob.squeeze(-1), batch.y.float())
                val_loss += loss.item() * batch.num_graphs
                
                # Calculate accuracy (Threshold = 0.5)
                preds = (zk_prob.squeeze(-1) > 0.5).float()
                correct += (preds == batch.y).sum().item()
                total_elements += batch.y.numel()
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = (correct / total_elements) * 100.0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Supervised Training for Erdős-GNN")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. LOAD EXCEL GRID DATA (Clean zero-capacity generators)
    case_path = f'../excel_outputs/{args.case}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)

    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses, zone2_buses = total_buses[:midpoint], total_buses[midpoint:]
    
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    global_kg = zonal_data['global_Kg']
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    num_boundaries, num_global_kg = len(boundary_buses), len(global_kg)
    bus_list = sorted(case['bus']['bus_i'].tolist())

    # 2. LOAD FEATURES (Generated Loads) AND LABELS (Master CSV)
    loads_df = pd.read_csv(f'data/{args.case}_generated_loads.csv')
    labels_df = pd.read_csv(f'data/labels/{args.case}_master_labels.csv')
    
    # Ensure they match in length
    num_instances = min(len(loads_df), len(labels_df))
    label_cols = [f"Gen_{k}_Active" for k in global_kg]

    dataset_z1, dataset_z2 = [], []

    print("Building PyTorch Geometric Datasets...")
    for s in range(num_instances):
        # Extract features (Loads in PU)
        scenario_row_pu = np.array([loads_df.iloc[s][f"Bus_{b}_Pd"] for b in bus_list]) / baseMVA
        load_z1 = scenario_row_pu[:len(zone1_buses)].reshape(1, -1)
        load_z2 = scenario_row_pu[len(zone1_buses):].reshape(1, -1)
        
        # FIXED: Extract target labels and unsqueeze so shape is [1, num_global_kg]
        y_vector = torch.tensor(labels_df.iloc[s][label_cols].values.astype(np.float32)).unsqueeze(0)

        # Create graphs and attach target tensor 'y'
        graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0]
        graph_z1.y = y_vector
        dataset_z1.append(graph_z1)

        graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0]
        graph_z2.y = y_vector
        dataset_z2.append(graph_z2)

    # 3. SPLIT DATA INTO TRAIN/VAL SETS (80/20)
    z1_train, z1_val = train_test_split(dataset_z1, test_size=0.2, random_state=42)
    z2_train, z2_val = train_test_split(dataset_z2, test_size=0.2, random_state=42)

    loader_z1_train = DataLoader(z1_train, batch_size=args.batch_size, shuffle=True)
    loader_z1_val = DataLoader(z1_val, batch_size=args.batch_size, shuffle=False)
    
    loader_z2_train = DataLoader(z2_train, batch_size=args.batch_size, shuffle=True)
    loader_z2_val = DataLoader(z2_val, batch_size=args.batch_size, shuffle=False)

    # 4. INITIALIZE MODELS
    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg)
    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg)

    # 5. TRAIN BOTH AGENTS
    print(f"Starting Supervised Training on {num_instances} scenarios...")
    gnn_z1 = train_agent("Zone 1 GNN", gnn_z1, loader_z1_train, loader_z1_val, device, num_global_kg, num_boundaries, args.epochs, args.lr)
    gnn_z2 = train_agent("Zone 2 GNN", gnn_z2, loader_z2_train, loader_z2_val, device, num_global_kg, num_boundaries, args.epochs, args.lr)

    # 6. SAVE WEIGHTS
    # Change the folder name to reflect the new architecture
    os.makedirs('data/ccga_models', exist_ok=True)
    
    save_z1 = f"data/ccga_models/zone1_gnn_oracle_{args.case}.pth"
    save_z2 = f"data/ccga_models/zone2_gnn_oracle_{args.case}.pth"
    
    torch.save(gnn_z1.state_dict(), save_z1)
    torch.save(gnn_z2.state_dict(), save_z2)
    
    print(f"\n=======================================================")
    print(f" SUCCESS! Models saved to:")
    print(f" - {save_z1}")
    print(f" - {save_z2}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()