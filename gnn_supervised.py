import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

def train_supervised_gnn(train_dataset, val_dataset, model, device, epochs=100, batch_size=32, lr=1e-3):
    """
    Trains the Erdős-GNN using exact CCGA ground-truth labels.
    """
    # 1. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Loss and Optimizer
    # Note: If your GNN outputs raw logits (no sigmoid at the end), use nn.BCEWithLogitsLoss() instead.
    # It is mathematically more stable than applying sigmoid + BCELoss.
    criterion = nn.BCELoss() 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.to(device)
    
    print("Starting Supervised Training...")
    
    for epoch in range(epochs):
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass (adjust dummy tensors based on your specific GNN inputs)
            va_dummy = torch.zeros(batch.num_graphs, model.num_global_kg + 1, model.num_boundaries, device=device)
            zk_dummy = torch.zeros(batch.num_graphs, model.num_global_kg, device=device)
            
            _, _, zk_prob = model(batch, va_dummy, va_dummy, zk_dummy, zk_dummy)
            
            # Calculate Binary Cross-Entropy Loss
            # batch.y must be a float tensor of shape (batch_size, num_global_kg) containing 0s and 1s
            loss = criterion(zk_prob.squeeze(), batch.y.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()
        total_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                va_dummy = torch.zeros(batch.num_graphs, model.num_global_kg + 1, model.num_boundaries, device=device)
                zk_dummy = torch.zeros(batch.num_graphs, model.num_global_kg, device=device)
                
                _, _, zk_prob = model(batch, va_dummy, va_dummy, zk_dummy, zk_dummy)
                loss = criterion(zk_prob.squeeze(), batch.y.float())
                total_val_loss += loss.item()
                
                # Compute multi-label accuracy using a 0.5 threshold
                predicted_classes = (zk_prob.squeeze() > 0.5).float()
                correct_predictions += (predicted_classes == batch.y).sum().item()
                total_predictions += batch.y.numel()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100
        
        # Print metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy:.2f}%")
            
    print("Training Complete!")
    return model