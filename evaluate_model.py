import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. Recreate the exact same architecture
class SupervisedContingencyGNN(nn.Module):
    def __init__(self, num_node_features, num_gen_targets, num_line_targets):
        super(SupervisedContingencyGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.gen_head = nn.Linear(128, num_gen_targets)
        self.line_head = nn.Linear(128, num_line_targets)

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return self.gen_head(h_graph), self.line_head(h_graph)

def evaluate(case_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Evaluating {case_name} on {device} ---")

    # Load the Test Dataset (the last 20%)
    data_dir = f"data/pyg_dataset/{case_name}"
    file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')])
    dataset = [torch.load(f, weights_only=False) for f in file_paths]
    test_dataset = dataset[int(0.8 * len(dataset)):]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model and load trained weights
    sample = dataset[0]
    model = SupervisedContingencyGNN(sample.x.shape[1], sample.y_gen.shape[0], sample.y_line.shape[0]).to(device)
    
    model_path = f"data/models/{case_name}_supervised_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_zeros = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            gen_logits, line_logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Combine gen and line predictions/targets for global metrics
            logits = torch.cat([gen_logits, line_logits], dim=1)
            targets = torch.cat([batch.y_gen.view(batch.num_graphs, -1), 
                                 batch.y_line.view(batch.num_graphs, -1)], dim=1)
            
            # Convert logits to binary predictions (Threshold = 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Tally up the results
            true_positives += ((preds == 1) & (targets == 1)).sum().item()
            false_positives += ((preds == 1) & (targets == 0)).sum().item()
            false_negatives += ((preds == 0) & (targets == 1)).sum().item()
            correct_zeros += ((preds == 0) & (targets == 0)).sum().item()

    # Calculate Metrics
    total_predictions = true_positives + false_positives + false_negatives + correct_zeros
    accuracy = (true_positives + correct_zeros) / total_predictions * 100
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print("\n--- TEST SET RESULTS (200 Scenarios) ---")
    print(f"Overall Accuracy:  {accuracy:.2f}% (Warning: Easily skewed by zeros)")
    print(f"Precision:         {precision * 100:.2f}% (When AI said 'Active', was it right?)")
    print(f"Recall:            {recall * 100:.2f}% (Did the AI catch all the real Active contingencies?)")
    print("----------------------------------------")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.case)