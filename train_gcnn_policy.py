import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

# Import the GCNN architecture from the previous script
# from gcnn_branching import MILP_LearnToBranch_GCNN

# =============================================================================
# 1. OFFLINE STRONG BRANCHING DATASET
# =============================================================================
class StrongBranchingDataset(Dataset):
    """
    Loads pre-collected bipartite graphs from solver logs.
    Each file should contain a PyG Data object with:
      - x_var, x_con, edge_index, edge_weight
      - candidate_mask: Boolean tensor of variables eligible for branching
      - target_policy: Normalized Strong Branching scores for the candidates
    """
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        return torch.load(file_path, weights_only=True)

# =============================================================================
# 2. IMITATION LEARNING LOSS FUNCTION
# =============================================================================
def compute_branching_loss(predicted_scores, target_policy, candidate_mask, batch_index):
    """
    Computes the Cross-Entropy loss between the GNN predictions and the 
    expert Strong Branching scores, masked to only consider eligible candidates.
    
    Formula: L = - sum( y_i * log( exp(s_i) / sum(exp(s_k)) ) )
    """
    loss = 0.0
    num_graphs = batch_index.max().item() + 1
    
    for g_idx in range(num_graphs):
        # Extract nodes belonging to the current graph in the batch
        graph_mask = (batch_index == g_idx)
        
        # Intersect with variables that are actually candidates for branching
        valid_candidates = graph_mask & candidate_mask
        
        if not valid_candidates.any():
            continue
            
        # Get raw logits and target probabilities for the candidates
        logits = predicted_scores[valid_candidates]
        targets = target_policy[valid_candidates]
        
        # Compute Log-Softmax over the candidates
        log_probs = F.log_softmax(logits, dim=0)
        
        # Cross Entropy: - sum(target * log_prob)
        graph_loss = -torch.sum(targets * log_probs)
        loss += graph_loss
        
    return loss / num_graphs

# =============================================================================
# 3. OFFLINE TRAINING ENGINE
# =============================================================================
def train_learn_to_branch(data_dir, model_save_path, epochs=50, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing Offline Learn-To-Branch Training on {device} ---")
    
    # 1. Load Dataset
    dataset = StrongBranchingDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} branching samples.")
    
    # 2. Initialize GNN & Optimizer
    # Assuming standard feature dimensions: 4 for vars, 2 for cons
    net = MILP_LearnToBranch_GCNN(var_in_dim=4, con_in_dim=2, hidden_dim=64).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # 3. Training Loop
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass: Predict unnormalized scores (logits) for all variables
            predicted_scores = net(
                batch.x_var, 
                batch.x_con, 
                batch.edge_index, 
                batch.edge_weight
            )
            
            # Calculate Cross-Entropy Loss
            loss = compute_branching_loss(
                predicted_scores, 
                batch.target_policy, 
                batch.candidate_mask, 
                batch.x_var_batch  # PyG automatically creates this batch mapping
            )
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Imitation Cross-Entropy Loss: {avg_loss:.4f}")
            
    # 4. Save Trained Policy
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(net.state_dict(), model_save_path)
    print(f"\n>>> Branching Policy Saved to: {model_save_path} <<<")

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Ensure you have a directory of .pt files containing the offline samples
    DATA_DIR = "data/strong_branching_samples/"
    SAVE_PATH = "data/admm_models/branch_gcnn_policy.pth"
    
    # Mock data generation for structural testing
    if not os.path.exists(DATA_DIR):
        print("Mocking data directory for testing...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for i in range(10):
            num_vars, num_cons = 50, 20
            mock_data = Data(
                x_var=torch.rand(num_vars, 4),
                x_con=torch.rand(num_cons, 2),
                edge_index=torch.randint(0, min(num_vars, num_cons), (2, 100)),
                edge_weight=torch.randn(100, 1),
                candidate_mask=(torch.rand(num_vars) > 0.5),
            )
            mock_data.target_policy = torch.rand(num_vars)
            mock_data.target_policy /= mock_data.target_policy.sum() # Normalize
            torch.save(mock_data, os.path.join(DATA_DIR, f"sample_{i}.pt"))
            
    train_learn_to_branch(DATA_DIR, SAVE_PATH, epochs=20, batch_size=4)