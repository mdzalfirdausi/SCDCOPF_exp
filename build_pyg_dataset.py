import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def build_pyg_dataset(case_name, features_path, labels_path, case_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading topology from {case_path}...")
    case_bus = pd.read_excel(case_path, sheet_name='bus')
    case_branch = pd.read_excel(case_path, sheet_name='branch')
    case_gen = pd.read_excel(case_path, sheet_name='gen')

    # 1. Build Graph Topology (edge_index)
    bus_list = sorted(case_bus['bus_i'].tolist())
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}

    edge_source = [bus_to_idx[i] for i in case_branch['bus_i'].values]
    edge_target = [bus_to_idx[j] for j in case_branch['bus_j'].values]
    
    # GNNs usually expect bidirectional edges for power grids
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)

    # 2. Extract Label Columns (to know which columns belong to Gen vs Line)
    print("Loading features and labels...")
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Ensure alignment (assuming Scenario_ID matches the row index of features_df)
    labels_df = labels_df.sort_values('Scenario_ID').reset_index(drop=True)
    
    gen_label_cols = [c for c in labels_df.columns if c.startswith('Gen_') and c.endswith('_Active')]
    line_label_cols = [c for c in labels_df.columns if c.startswith('Line_') and c.endswith('_Active')]

    print(f"Found {len(features_df)} scenarios, {len(gen_label_cols)} gen targets, {len(line_label_cols)} line targets.")

    # 3. Build and Save PyG Graphs
    success_count = 0
    
    # CORRECTED UNPACKING: (index1, row1), (index2, row2)
    for (idx, feature_row), (_, label_row) in tqdm(zip(features_df.iterrows(), labels_df.iterrows()), total=len(features_df)):
        try:
            # --- NODE FEATURES (x) ---
            # Directly access the column from feature_row
            load_features = np.array([feature_row[f"Bus_{b}_Pd"] for b in bus_list], dtype=np.float32)
            x_tensor = torch.tensor(load_features).view(-1, 1)

            # --- TARGETS (y) ---
            # Directly access the column from label_row
            y_gen = torch.tensor([label_row[col] for col in gen_label_cols], dtype=torch.float32)
            y_line = torch.tensor([label_row[col] for col in line_label_cols], dtype=torch.float32)

            # --- BUILD GRAPH DATA OBJECT ---
            graph_data = Data(
                x=x_tensor, 
                edge_index=edge_index, 
                y_gen=y_gen,    
                y_line=y_line   
            )

            # --- SAVE TO DISK ---
            scenario_id = int(label_row['Scenario_ID'])
            torch.save(graph_data, os.path.join(output_dir, f"scenario_{scenario_id}.pt"))
            success_count += 1

        except Exception as e:
            print(f"Error on scenario {idx}: {e}")

    print(f"\nSuccessfully generated {success_count} PyG graphs in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG Dataset from Features and Labels")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case300_ieee")
    args = parser.parse_args()

    # Define paths
    case_name = args.case
    features_path = f"data/{case_name}_generated_data.csv" 
    labels_path = f"data/labels/ccga_{case_name}/{case_name}_master_labels.csv"         
    case_path = f"../excel_outputs/{case_name}.xlsx"
    output_dir = f"data/pyg_dataset/{case_name}"

    build_pyg_dataset(case_name, features_path, labels_path, case_path, output_dir)