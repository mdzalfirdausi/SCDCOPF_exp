import os
import sys

# ADD THIS LINE BEFORE ANY OTHER IMPORTS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# Import your custom physical solver
from dcopf_model2 import build_ptdf, run_ccga_algorithm

def generate_ccga_dataset(bus_df, gen_df, branch_df, cost_df, loads_df, PTDF_matrix, start_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Map generator IDs to indices for building the binary target tensor
    gen_list = gen_df['gen_ID'].tolist()
    gen_to_idx = {gen_id: idx for idx, gen_id in enumerate(gen_list)}
    
    # Extract topology for the GNN
    bus_list = sorted(bus_df['bus_i'].tolist())
    bus_to_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    
    # Create the edge_index mapping
    edge_source = [bus_to_idx[i] for i in branch_df['bus_i'].values]
    edge_target = [bus_to_idx[j] for j in branch_df['bus_j'].values]
    edge_index = torch.tensor([edge_source + edge_target, edge_target + edge_source], dtype=torch.long)

    success_count = 0
    
    # Execute loop over EVERY load in the dataframe
    for local_idx, row in loads_df.reset_index(drop=True).iterrows():
        global_idx = start_idx + local_idx
        
        # Build load_vector specific to this instance
        load_vector = {bus_id: row[f"Bus_{bus_id}_Pd"] for bus_id in bus_list}
        
        try:
            print(f"Solving Instance {global_idx}...")
            
            # RUN EXACT SOLVER
            optimal_g, status, iters, active_S = run_ccga_algorithm(
                bus_df, gen_df, branch_df, cost_df, load_vector, PTDF_matrix  
            )
            
            # Create the Binary Label (1 if contingency is in active_S, 0 otherwise)
            y_target = np.zeros(len(gen_list), dtype=np.float32)
            for s in active_S:
                if s in gen_to_idx:
                    y_target[gen_to_idx[s]] = 1.0
                    
            # Create the Input Features (Nodal Load)
            load_features = np.array([load_vector[b] for b in bus_list], dtype=np.float32)
            
            # Pack into a PyTorch Geometric Data object
            graph_data = Data(
                x=torch.tensor(load_features).view(-1, 1), # Node features: [Num_buses, 1]
                edge_index=edge_index,                     # Graph topology
                y=torch.tensor(y_target)                   # Target labels: [Num_gens]
            )
            
            # Save PyG graph to disk
            torch.save(graph_data, os.path.join(output_dir, f"scenario_{global_idx}.pt"))
            success_count += 1
            
            print(f"  -> Instance {global_idx} processed. Active contingencies: {len(active_S)}")
            
        except Exception as e:
            print(f"Instance {global_idx} failed: {e}")

    print(f"\nSuccessfully generated {success_count} CCGA PyG samples.")


def main():
    # 1. Set up argparse to accept the --case flag
    parser = argparse.ArgumentParser(description="Generate PyG CCGA Dataset for a specific power system case.")
    parser.add_argument('--case', type=str, required=True, 
                        help="The name of the case to run (e.g., 'pglib_opf_case118_ieee')")
    args = parser.parse_args()
    
    case_name = args.case

    # 2. Hardcode start_idx to 0 since we are processing the whole file at once
    start_idx = 0

    # 3. Define accurate relative paths dynamically based on the case
    base_data_dir = "../excel_outputs/"
    gen_data_dir = "./data/"
    
    excel_path = os.path.join(base_data_dir, f"{case_name}.xlsx")
    loads_path = os.path.join(gen_data_dir, f"{case_name}_generated_loads.csv")
    output_dir = os.path.join(gen_data_dir, "ccga_samples", case_name)

    # 4. Load Base Network Data
    print(f"Loading network topology for case '{case_name}' from {excel_path}...")
    bus_df = pd.read_excel(excel_path, sheet_name='bus')
    gen_df = pd.read_excel(excel_path, sheet_name='gen')
    branch_df = pd.read_excel(excel_path, sheet_name='branch')
    cost_df = pd.read_excel(excel_path, sheet_name='gencost')
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    
    print(f"Pre-computing PTDF Matrix (Reference Bus: {ref_bus_id})...")
    PTDF_matrix, bus_list = build_ptdf(bus_df, branch_df, ref_bus_id)

    # 5. Load the ENTIRE generated loads file
    print(f"Loading ALL instances from {loads_path}...")
    loads_df = pd.read_csv(loads_path)
    print(f"Detected {len(loads_df)} load scenarios to process.")
    
    # 6. Call the dataset generator
    generate_ccga_dataset(
        bus_df=bus_df, 
        gen_df=gen_df, 
        branch_df=branch_df, 
        cost_df=cost_df,
        loads_df=loads_df,
        PTDF_matrix=PTDF_matrix,
        start_idx=start_idx,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()