import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ==========================================
# 1. SUPERVISED GNN ARCHITECTURE
# ==========================================
class SupervisedContingencyGNN(nn.Module):
    def __init__(self, num_node_features, num_gen_targets, num_line_targets):
        super(SupervisedContingencyGNN, self).__init__()
        
        # Graph Convolutional Layers (Learn spatial relationships)
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        
        # Output Heads (Predict contingencies from the graph embedding)
        self.gen_head = nn.Linear(128, num_gen_targets)
        self.line_head = nn.Linear(128, num_line_targets)

    def forward(self, x, edge_index, batch):
        # Pass features through GCN layers
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        
        # Pool all nodes together to get a single vector for the whole grid
        h_graph = global_mean_pool(h, batch)
        
        # Output raw prediction logits (probabilities are calculated in the loss function)
        gen_logits = self.gen_head(h_graph)
        line_logits = self.line_head(h_graph)
        
        return gen_logits, line_logits

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_model(case_name, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Supervised Training on {device} ---")

    # Load all the .pt files into memory
    data_dir = f"data/pyg_dataset/{case_name}"
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    print(f"Loading {len(file_paths)} PyG graphs...")
    dataset = [torch.load(f, weights_only=False) for f in file_paths]
    
    # Split Dataset: 80% for training, 20% for testing
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Automatically detect input/output sizes from the first graph
    sample = dataset[0]
    num_node_features = sample.x.shape[1]      
    num_gen_targets = sample.y_gen.shape[0]    # e.g., 57
    num_line_targets = sample.y_line.shape[0]  # e.g., 411

    # Initialize the Model, Optimizer, and Loss Function
    model = SupervisedContingencyGNN(num_node_features, num_gen_targets, num_line_targets).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # BCEWithLogitsLoss is the standard for multi-label binary classification
    criterion = nn.BCEWithLogitsLoss() 

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            gen_logits, line_logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate Loss (compare GNN guess vs Exact Solver target)
            loss_gen = criterion(gen_logits, batch.y_gen.view(-1, num_gen_targets))
            loss_line = criterion(line_logits, batch.y_line.view(-1, num_line_targets))
            
            # Backpropagation
            loss = loss_gen + loss_line
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Training Loss: {total_loss/len(train_loader):.4f}")

    # Save the finalized model
    os.makedirs("data/models", exist_ok=True)
    save_path = f"data/models/{case_name}_supervised_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supervised GNN")
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_model(args.case, args.epochs, args.batch_size)