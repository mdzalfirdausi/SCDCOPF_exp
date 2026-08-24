import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# =============================================================================
# 1. GRAPH NEURAL NETWORK ARCHITECTURE
# =============================================================================
class ErdosGridGNN(nn.Module):
    """
    Graph Neural Network that outputs Bernoulli parameters (probabilities)
    p_i in [0, 1] for each discrete candidate decision (e.g., line switching,
    breaker status, or generator commitment).
    """
    def __init__(self, in_channels=4, hidden_channels=64, num_heads=4):
        super(ErdosGridGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        
        # Decision Head: maps node embeddings to Bernoulli probabilities p_i in [0, 1]
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid() # Outputs Bernoulli probabilities
        )

    def forward(self, x, edge_index):
        h = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        h = F.leaky_relu(self.conv2(h, edge_index), 0.1)
        
        # Probabilities p for each node/component
        probs = self.prob_head(h).squeeze(-1)
        return probs


# =============================================================================
# 2. DIFFERENTIABLE ERDŐS PROBABILISTIC LOSS
# =============================================================================
def erdos_probabilistic_loss(probs, edge_index, x_features, target_k=5, beta_budget=10.0, beta_conn=15.0):
    """
    Computes: Loss = E[Cost] + beta * P(Constraints Violated)
    
    Using Bernoulli expectations:
      - E[z_i] = p_i
      - E[z_i * z_j] = p_i * p_j (for independent Bernoulli variables)
      - E[(sum z_i - K)^2] = sum(p_i * (1 - p_i)) + (sum(p_i) - K)^2
    """
    # --- 1. Expected Operational Cost E[Cost(S)] ---
    # Example: Linear generation/dispatch cost coefficient extracted from node features
    cost_weights = x_features[:, 0] # e.g., cost coefficient per unit
    expected_cost = torch.sum(cost_weights * probs)

    # --- 2. Cardinality / Budget Constraint Penalty ---
    # Penalizes deviation from desired number of active components (target_k)
    # E[(sum(z_i) - K)^2] = Var(sum(z_i)) + (E[sum(z_i)] - K)^2
    expected_sum = torch.sum(probs)
    variance_sum = torch.sum(probs * (1.0 - probs))
    expected_budget_violation = variance_sum + torch.square(expected_sum - target_k)

    # --- 3. Topology Connectivity / Flow Interaction Penalty ---
    # Penalizes pairs of connected components disconnected across active boundary
    src, dst = edge_index
    # E[z_i * (1 - z_j)] = p_i * (1 - p_j)
    expected_boundary_mismatch = torch.sum(probs[src] * (1.0 - probs[dst]))

    # Total Differentiable Probabilistic Loss
    total_loss = (
        expected_cost 
        + beta_budget * expected_budget_violation 
        + beta_conn * expected_boundary_mismatch
    )
    return total_loss


# =============================================================================
# 3. METHOD OF CONDITIONAL EXPECTATION (SEQUENTIAL DECODER)
# =============================================================================
def decode_conditional_expectation(probs, edge_index, x_features, target_k=5, beta_budget=10.0, beta_conn=15.0):
    """
    Derandomizes the learned probabilities into hard binary variables {0, 1}
    guaranteed to satisfy the probabilistic bound.
    """
    with torch.no_grad():
        p_current = probs.clone()
        num_vars = p_current.shape[0]

        # Step 5a: Sort indices by probability in descending order
        sorted_indices = torch.argsort(p_current, descending=True)

        # Step 5b: Sequentially evaluate conditional expectations
        for idx in sorted_indices:
            # Option 1: Fix variable to 1
            p_current[idx] = 1.0
            loss_one = erdos_probabilistic_loss(
                p_current, edge_index, x_features, target_k, beta_budget, beta_conn
            ).item()

            # Option 2: Fix variable to 0
            p_current[idx] = 0.0
            loss_zero = erdos_probabilistic_loss(
                p_current, edge_index, x_features, target_k, beta_budget, beta_conn
            ).item()

            # Lock in the discrete choice that minimizes conditional expected loss
            if loss_one <= loss_zero:
                p_current[idx] = 1.0
            else:
                p_current[idx] = 0.0

        # Exact discrete integral solution vector
        discrete_solution = p_current.to(torch.int64)
        return discrete_solution


# =============================================================================
# 4. TRAINING & INFERENCE WORKFLOW
# =============================================================================
def generate_synthetic_grid(num_nodes=30, num_edges=50):
    """Generates a dummy PyG Data object simulating power grid features."""
    x = torch.rand((num_nodes, 4), dtype=torch.float32) # [Cost, P_load, P_max, P_min]
    edge_src = torch.randint(0, num_nodes, (num_edges,))
    edge_dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    # Initialize Model & Optimizer
    model = ErdosGridGNN(in_channels=4, hidden_channels=64, num_heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Generate synthetic training dataset (Unsupervised, no labels needed)
    train_graphs = [generate_synthetic_grid().to(device) for _ in range(50)]
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)

    print("\n--- Starting Unsupervised Erdős Training ---")
    model.train()
    for epoch in range(1, 21):
        total_epoch_loss = 0.0
        for graph in train_loader:
            optimizer.zero_grad()
            
            # Forward pass -> Bernoulli probabilities
            probs = model(graph.x, graph.edge_index)
            
            # Compute Unsupervised Probabilistic Loss
            loss = erdos_probabilistic_loss(
                probs, graph.edge_index, graph.x, 
                target_k=6, beta_budget=12.0, beta_conn=8.0
            )
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/20] | Unsupervised Loss: {total_epoch_loss / len(train_graphs):.4f}")

    print("\n--- Inference: Sequential Decoding via Method of Conditional Expectation ---")
    model.eval()
    test_graph = generate_synthetic_grid().to(device)
    
    with torch.no_grad():
        predicted_probs = model(test_graph.x, test_graph.edge_index)

    # Perform deterministic sequential decoding
    discrete_decisions = decode_conditional_expectation(
        predicted_probs, test_graph.edge_index, test_graph.x, 
        target_k=6, beta_budget=12.0, beta_conn=8.0
    )

    print(f"Predicted Continuous Probabilities (First 10):\n{predicted_probs[:10].cpu().numpy().round(3)}")
    print(f"Decoded Discrete Binary Decisions  (First 10):\n{discrete_decisions[:10].cpu().numpy()}")
    print(f"Total Components Activated: {discrete_decisions.sum().item()} (Target was 6)")