import pandas as pd
import numpy as np
import os
import argparse

def generate_datasets(case_name, num_instances):
    # 1. Set the seed for exact reproducibility
    np.random.seed(42)
    
    filepath = f"../excel_outputs/{case_name}.xlsx"
    output_dir = r"./data"
    
    print(f"Loading base network data from {filepath}...")
    bus_df = pd.read_excel(filepath, sheet_name='bus')
    gen_df = pd.read_excel(filepath, sheet_name='gen')
    cost_df = pd.read_excel(filepath, sheet_name='gencost')

    # =====================================================================
    # 1. LOAD DEMAND PERTURBATION (Truncated Multivariate Gaussian)
    # =====================================================================
    d_0 = bus_df['Pd'].values
    num_buses = len(d_0)
    mu_load = 0.5  # 50% max perturbation as defined in the 2025 paper
    
    # Scale sigma by the Z-score for the 95th percentile (approx 1.96)
    sigma_d = (mu_load * d_0) / 1.96
    
    # Build Covariance Matrix for Loads (alpha = 0.5 for different buses)
    Sigma_d = np.zeros((num_buses, num_buses))
    for i in range(num_buses):
        for j in range(num_buses):
            if i == j:
                Sigma_d[i, j] = sigma_d[i]**2
            else:
                Sigma_d[i, j] = 0.5 * sigma_d[i] * sigma_d[j]

    print(f"Sampling {num_instances} load demands with spatial correlation...")
    # Sample from multivariate normal
    sampled_loads = np.random.multivariate_normal(d_0, Sigma_d, num_instances)
    
    # Apply strict truncation limits: (1-mu)d_0 and (1+mu)d_0
    lower_bound_d = (1 - mu_load) * d_0
    upper_bound_d = (1 + mu_load) * d_0
    sampled_loads = np.clip(sampled_loads, lower_bound_d, upper_bound_d)

    # =====================================================================
    # 2. GENERATOR LIMIT & COST PERTURBATION
    # =====================================================================
    num_gens = len(gen_df)
    std_factor = 0.1 # Standard deviation for the multiplier factors
    sigma_g = np.full(num_gens, std_factor)
    
    # Build Covariance Matrix for Generators (alpha = 0.8 for different gens)
    Sigma_g = np.zeros((num_gens, num_gens))
    for i in range(num_gens):
        for j in range(num_gens):
            if i == j:
                Sigma_g[i, j] = sigma_g[i]**2
            else:
                Sigma_g[i, j] = 0.8 * sigma_g[i] * sigma_g[j]

    print("Sampling generator capacities and cost coefficients...")
    # Two independent factors sampled from N(1, Sigma)
    factors_Pmax = np.random.multivariate_normal(np.ones(num_gens), Sigma_g, num_instances)
    factors_cost = np.random.multivariate_normal(np.ones(num_gens), Sigma_g, num_instances)

    # Extract base physical parameters
    Pmax_base = gen_df['Pmax'].values
    Pmin_base = gen_df['Pmin'].values
    g_hat = Pmax_base - Pmin_base
    
    c1_base = cost_df['c1'].values
    c2_base = cost_df['c2'].values

    # =====================================================================
    # 3. BUILD THE COMPREHENSIVE DATASET
    # =====================================================================
    dataset_rows = []
    
    for k in range(num_instances):
        row_dict = {}
        
        # Insert perturbed loads
        for idx, bus_id in enumerate(bus_df['bus_i']):
            row_dict[f"Bus_{bus_id}_Pd"] = sampled_loads[k, idx]

        # Insert perturbed generator limits and costs
        for idx, gen_id in enumerate(gen_df['gen_ID']):
            
            # Safeguard 1: Costs must be non-negative
            new_c1 = max(c1_base[idx] * factors_cost[k, idx], 0.0)
            new_c2 = max(c2_base[idx] * factors_cost[k, idx], 0.0)
            
            # Safeguard 2: Pmax must be at least Pmin + 1% of total capacity
            new_Pmax = max(Pmax_base[idx] * factors_Pmax[k, idx], Pmin_base[idx] + 0.01 * g_hat[idx])
            
            row_dict[f"Gen_{gen_id}_Pmax"] = new_Pmax
            row_dict[f"Gen_{gen_id}_c1"] = new_c1
            row_dict[f"Gen_{gen_id}_c2"] = new_c2

        dataset_rows.append(row_dict)

    dataset_df = pd.DataFrame(dataset_rows)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{case_name}_generated_data.csv")
    dataset_df.to_csv(output_filename, index=False)
    print(f"\nSuccess: Exported {dataset_df.shape[0]} comprehensive scenarios to {output_filename}")

    return dataset_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive ML datasets (Loads, Limits, Costs).")
    parser.add_argument('--case', type=str, required=True, 
                        help="The name of the case to run (e.g., 'pglib_opf_case118_ieee')")
    parser.add_argument('--num_instances', type=int, default=1000, 
                        help="Number of profiles to generate (default: 1000)")
    args = parser.parse_args()
    
    filepath = f"../excel_outputs/{args.case}.xlsx"
    if os.path.exists(filepath):
        generate_datasets(args.case, args.num_instances)
    else:
        print(f"Error: {filepath} not found. Please ensure the Excel file exists.")