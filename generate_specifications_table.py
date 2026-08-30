import os
import argparse
import pandas as pd

def extract_case_specifications(case_name, base_dir="../excel_outputs/"):
    case_path = os.path.join(base_dir, f"{case_name}.xlsx")
    
    # 1. Load the Excel Data
    case = pd.read_excel(case_path, sheet_name=['baseMVA', 'bus', 'gen', 'branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    # 2. Clean generators exactly as done in evaluate_benchmark_2.py
    gen_df = case['gen'].copy()
    pmax_pu = gen_df['Pmax'].values / baseMVA
    pmin_pu = gen_df['Pmin'].values / baseMVA
    
    zero_gen_idx = [
        num for num, pmax in enumerate(pmax_pu) 
        if (pmax == 0 and pmin_pu[num] == 0) or pmin_pu[num] < 0
    ]
    gen_df.drop(index=zero_gen_idx, inplace=True)
    
    # 3. Calculate Table Metrics
    N_buses = len(case['bus'])                               # |N| Nodes/Buses
    G_gens = len(gen_df)                                     # |G| Generators
    L_loads = (case['bus']['Pd'] > 0).sum()                  # |L| Active Loads
    E_branches = len(case['branch'])                         # |E| Edges/Branches
    Kg_contingencies = len(gen_df)                           # |Kg| Gen Contingencies
    
    # Note: dim(x) varies by exact OPF formulation. 
    # For a standard DC-OPF state vector, it is usually (N_buses - 1) + G_gens
    dim_x = (N_buses - 1) + G_gens                           

    return {
        "Test Case": case_name,
        "$|\mathcal{N}|$": N_buses,
        "$|\mathcal{G}|$": G_gens,
        "$|\mathcal{L}|$": L_loads,
        "$|\mathcal{E}|$": E_branches,
        "$|\mathcal{K}_g|$": Kg_contingencies,
        "dim(x)": dim_x
    }

if __name__ == "__main__":
    # Set up argparse to accept the --case flag
    parser = argparse.ArgumentParser(description="Generate Specifications Table for SCOPF Test Cases")
    
    # nargs='+' allows you to pass one or multiple cases separated by spaces
    parser.add_argument('--case', type=str, nargs='+', required=True, 
                        help="The name(s) of the case(s) to process (e.g., --case pglib_opf_case14_ieee pglib_opf_case118_ieee)")
    
    args = parser.parse_args()
    
    results = []
    for case_name in args.case:
        try:
            stats = extract_case_specifications(case_name)
            results.append(stats)
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
            
    if results:
        # Compile into a DataFrame
        df_results = pd.DataFrame(results)
        
        print("\n=== CONSOLE OUTPUT ===")
        print(df_results.to_string(index=False))
        
        print("\n=== LATEX TABLE OUTPUT ===")
        # Generates the exact LaTeX code for your paper
        latex_code = df_results.to_latex(
            index=False, 
            escape=False, 
            column_format="lcccccc"
        )
        print(latex_code)
    else:
        print("No valid cases were processed.")