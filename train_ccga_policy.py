import os
import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, global_mean_pool

# =============================================================================
# 1. LOAD CCGA DATASET
# =============================================================================
class CCGADataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.data_dir, self.file_names[idx]), weights_only=False)

# =============================================================================
# 2. GNN CONTINGENCY PREDICTOR
# =============================================================================
class FastCCGAPredictor(nn.Module):
    def __init__(self, num_gens, hidden_dim=64):
        super().__init__()
        # Input dimension is 1 (Nodal active power load, Pd)
        self.conv1 = GATConv(1, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        
        # Output exactly `num_gens` logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gens)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message Passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Aggregate the entire grid into one embedding vector
        x_graph = global_mean_pool(x, batch)
        
        return self.classifier(x_graph)

# =============================================================================
# 3. TRAINING ENGINE
# =============================================================================
def train_ccga_predictor(data_dir, model_save_path, num_gens, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training CCGA Active_S Predictor on {device} ---")
    
    dataset = CCGADataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} grid scenarios.")
    
    model = FastCCGAPredictor(num_gens=num_gens).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Binary Cross-Entropy Loss for multi-label classification (0 or 1 for each generator)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch)
            
            # Reshape target [Batch_Size * num_gens] -> [Batch_Size, num_gens]
            target = batch.y.view(-1, num_gens)
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | BCE Loss: {avg_loss:.4f}")
            
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n>>> GNN Policy Saved to: {model_save_path} <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CCGA Surrogate GNN")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    args = parser.parse_args()
    
    DATA_DIR = f"data/ccga_samples/{args.case}/"
    SAVE_PATH = f"data/admm_models/ccga_gnn_policy_{args.case}.pth"
    EXCEL_PATH = f"../excel_outputs/{args.case}.xlsx"
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Run generate_ccga_dataset.py first.")
    elif not os.path.exists(EXCEL_PATH):
        print(f"Error: Base grid data not found at {EXCEL_PATH}.")
    else:
        # Dynamically determine the number of generators from the dataset
        gen_df = pd.read_excel(EXCEL_PATH, sheet_name='gen')
        num_generators = len(gen_df)
        print(f"Detected {num_generators} generators for case '{args.case}'.")
        
        train_ccga_predictor(DATA_DIR, SAVE_PATH, num_gens=num_generators)