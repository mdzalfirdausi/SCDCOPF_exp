import pandas as pd
import numpy as np
import os

def generate_load_profiles(case, num_instances=1000):
    # 1. Set the seed for exact reproducibility
    np.random.seed(42)
    filepath = f"../excel_outputs/{case}.xlsx"
    output_dir = r"./data"
    print("Loading base network data...")
    bus_df = pd.read_excel(filepath, sheet_name='bus')
    
    base_Pd = bus_df['Pd'].values
    num_buses = len(base_Pd)
    
    load_matrix = np.zeros((num_instances, num_buses))
    
    print(f"Generating {num_instances} load instances...")
    
    for i in range(num_instances):
        deterministic_scale = 0.82 + (i * 0.00002)
        deterministic_Pd = base_Pd * deterministic_scale
        
        # Because the seed is set, this uniform distribution will 
        # yield the exact same arrays across every script execution
        random_noise_scale = np.random.uniform(-0.005, 0.005, size=num_buses)
        random_Pd = base_Pd * random_noise_scale
        
        load_matrix[i, :] = deterministic_Pd + random_Pd

    column_names = [f"Bus_{bus_id}_Pd" for bus_id in bus_df['bus_i']]
    dataset_df = pd.DataFrame(load_matrix, columns=column_names)
    
    # 2. Route the output to your mapped Explorer drive
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{case}.csv")
    
    dataset_df.to_csv(output_filename, index=False)
    print(f"Success: Exported exactly {dataset_df.shape[0]} summarized rows to {output_filename}")
    
    return dataset_df

if __name__ == "__main__":
    case = "pglib_opf_case1354_pegase"
    filepath = f"../excel_outputs/{case}.xlsx"
    if os.path.exists(filepath):
        generated_data = generate_load_profiles(case)
    else:
        print(f"Error: {filepath} not found in the current directory.")