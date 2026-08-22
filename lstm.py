import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. PYTORCH DATASET DEFINITION
# ==========================================
class ADMMTrajectoryDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        X_data: numpy array of shape [num_samples, seq_length, num_features]
                (The trajectory of multipliers/primals for early iterations)
        y_data: numpy array of shape [num_samples, num_features]
                (The final converged multipliers/primals)
        """
        # Convert data to PyTorch tensors
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. LSTM MODEL ARCHITECTURE
# ==========================================
class ADMM_Accelerator_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(ADMM_Accelerator_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer to capture the trajectory dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layers to map the hidden state to the final converged values
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        
        # Initialize hidden and cell states with zeros (handled automatically if not provided, 
        # but good practice for clarity on HPC nodes)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Out shape: [batch_size, seq_length, hidden_dim]
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We only care about the output from the final time step of the sequence
        final_time_step_out = out[:, -1, :] 
        
        # Pass through the fully connected layers
        x = self.fc1(final_time_step_out)
        x = self.relu(x)
        predictions = self.fc2(x)
        
        return predictions

# ==========================================
# 3. TRAINING LOOP 
# ==========================================
def train_model(model, dataloader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss() # Mean Squared Error is perfect for continuous OPF variables
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Compute loss
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Example Execution
if __name__ == "__main__":
    # Ensure PyTorch uses the powerful GPUs available on the cluster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # ---------------------------------------------------------
    # TODO: Load your extracted ADMM matrices here!
    # For now, we will create dummy data to verify the architecture compiles.
    # Suppose we track 50 features (u and Va) over the first 10 iterations.
    num_samples = 14000 # Your generated load profiles
    seq_length = 10     # Observing iterations 1 through 10
    num_features = 50   # Flattened size of u_va, u_zk, Va_base, zk
    
    dummy_X = np.random.randn(num_samples, seq_length, num_features)
    dummy_y = np.random.randn(num_samples, num_features) # The converged values
    # ---------------------------------------------------------
    
    dataset = ADMMTrajectoryDataset(dummy_X, dummy_y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ADMM_Accelerator_LSTM(
        input_dim=num_features, 
        hidden_dim=128, 
        num_layers=2, 
        output_dim=num_features
    )
    
    train_model(model, dataloader, num_epochs=100, learning_rate=0.001, device=device)