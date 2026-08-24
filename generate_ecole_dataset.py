import os
import ecole
import numpy as np
import torch
from torch_geometric.data import Data

def generate_strong_branching_dataset(mps_file_path, output_dir):
    """
    Runs an episode of Branch-and-Bound using SCIP via Ecole.
    Collects the bipartite graph observation and Strong Branching scores at each node.
    """
    # 1. Define the Ecole Environment
    # We request Bipartite observations and Strong Branching score information
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={"expert_scores": ecole.information.StrongBranchingScores()}
    )

    # 2. Configure SCIP to act as the Expert
    # These parameters force SCIP to run Full Strong Branching without limits
    scip_parameters = {
        "branching/allfullstrong/priority": 9999999,
        "branching/allfullstrong/maxdepth": -1
    }

    # 3. Reset the environment with your MILP instance
    # Ecole reads .mps or .lp files natively
    obs, action_set, reward, done, info = env.reset(mps_file_path, scip_parameters)
    
    node_counter = 0

    # 4. Step through the Branch-and-Bound tree
    while not done:
        # The observation contains the bipartite graph features
        row_features = obs.row_features      # Constraint features
        col_features = obs.column_features   # Variable features
        edge_indices = obs.edge_features.indices # Bipartite connections
        edge_values = obs.edge_features.values
        
        # The info dictionary contains the exact strong branching scores calculated by SCIP
        # Note: Ecole returns scores for the fractional candidates (action_set)
        sb_scores = info["expert_scores"]
        
        # Normalize the scores to create a target probability distribution
        target_policy = np.zeros(len(col_features))
        if sb_scores.sum() > 0:
            target_policy[action_set] = sb_scores / sb_scores.sum()
        else:
            target_policy[action_set] = 1.0 / len(action_set) # Uniform fallback
            
        candidate_mask = np.zeros(len(col_features), dtype=bool)
        candidate_mask[action_set] = True

        # 5. Pack into a PyTorch Geometric Data object
        graph_data = Data(
            x_var=torch.tensor(col_features, dtype=torch.float32),
            x_con=torch.tensor(row_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.long),
            edge_weight=torch.tensor(edge_values, dtype=torch.float32).unsqueeze(-1),
            target_policy=torch.tensor(target_policy, dtype=torch.float32),
            candidate_mask=torch.tensor(candidate_mask, dtype=torch.bool)
        )
        
        torch.save(graph_data, f"{output_dir}/node_{node_counter}.pt")
        node_counter += 1

        # 6. Step the environment forward using the Expert's choice
        # We pick the variable with the highest Strong Branching score
        expert_action = action_set[np.argmax(sb_scores)]
        obs, action_set, reward, done, info = env.step(expert_action)
        
    print(f"Collected {node_counter} branching samples from {mps_file_path}")

# Example Usage:
# generate_strong_branching_dataset("admm_subproblem_zone1.mps", "data/sb_samples/")