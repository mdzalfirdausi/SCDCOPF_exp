import os
import time
import argparse
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

# =====================================================================
# 1. BENDERS MASTER PROBLEM
# =====================================================================
def build_benders_master(bus_df, gen_df, branch_df, cost_df, load_vector, baseMVA=100.0):
    """
    Builds the Master Problem (Eq 22-24).
    Contains nominal scheduling AND the complete primary response logic for all contingencies.
    """
    model = pyo.ConcreteModel()
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    model.Contingencies = pyo.Set(initialize=model.Gens) 
    
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    c1 = dict(zip(cost_df['gen_ID'], cost_df['c1']))
    c2 = dict(zip(cost_df['gen_ID'], cost_df['c2']))
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    
    gamma_val = gen_df.attrs.get('gamma', 0.05)
    gamma = {i: gamma_val for i in model.Gens}
    g_hat = {i: pmax[i] - pmin[i] for i in model.Gens}

    bus_gens = {b: [] for b in model.Buses}
    for _, row in gen_df.iterrows():
        bus_gens[row['bus_i']].append(row['gen_ID'])
        
    lines_from = {b: [] for b in model.Buses}
    lines_to = {b: [] for b in model.Buses}
    branch_ends = {}
    for _, row in branch_df.iterrows():
        lines_from[row['bus_i']].append(row['line_ID'])
        lines_to[row['bus_j']].append(row['line_ID'])
        branch_ends[row['line_ID']] = (row['bus_i'], row['bus_j'])

    # Variables: Nominal State
    model.Pg = pyo.Var(model.Gens) 
    model.Theta = pyo.Var(model.Buses, bounds=(-np.pi, np.pi))
    model.Pf = pyo.Var(model.Branches)

    # Variables: Contingency States (Master Problem handles all APR binary variables)
    model.Pgs = pyo.Var(model.Contingencies, model.Gens, within=pyo.NonNegativeReals)
    model.ns = pyo.Var(model.Contingencies, bounds=(0, 1))
    model.xs = pyo.Var(model.Contingencies, model.Gens, domain=pyo.Binary)

    # Objective: Minimize nominal cost
    def obj_rule(m):
        return sum(c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints: Nominal Power Flow
    def flow_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf[l] == (m.Theta[bus_from] - m.Theta[bus_to]) / x[l]
    model.flow_eq = pyo.Constraint(model.Branches, rule=flow_rule)

    def limit_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return (-rateA[l], m.Pf[l], rateA[l])
    model.limit_eq = pyo.Constraint(model.Branches, rule=limit_rule)

    def pg_bound_rule(m, i):
        return (pmin[i], m.Pg[i], pmax[i])
    model.pg_bound_eq = pyo.Constraint(model.Gens, rule=pg_bound_rule)

    def balance_rule(m, b):
        gen_total = sum(m.Pg[g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        return gen_total - load_total == sum(m.Pf[l] for l in lines_from[b]) - sum(m.Pf[l] for l in lines_to[b])
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)
    model.ref_bus = pyo.Constraint(expr=model.Theta[ref_bus_id] == 0)

    # Constraints: Post-Contingency Generation 
    def failed_gen_rule(m, s):
        return m.Pgs[s, s] == 0
    model.failed_gen_eq = pyo.Constraint(model.Contingencies, rule=failed_gen_rule)

    def global_demand_rule(m, s):
        total_load = sum(load_vector[b] for b in m.Buses) / baseMVA
        return sum(m.Pgs[s, i] for i in m.Gens) == total_load
    model.global_demand_eq = pyo.Constraint(model.Contingencies, rule=global_demand_rule)

    # APR Logic (Big-M formulation)
    def apr_rule1(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * g_hat[i] <= pmax[i] * (1 - m.xs[s, i])
    model.apr_eq1 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_rule1)

    def apr_rule2(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * g_hat[i] >= -pmax[i] * (1 - m.xs[s, i])
    model.apr_eq2 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_rule2)

    def apr_rule3(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pg[i] + m.ns[s] * gamma[i] * g_hat[i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_eq3 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_rule3)

    def apr_rule4(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs[s, i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_eq4 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_rule4)

    # Empty ConstraintList to dynamically catch Benders cuts
    model.benders_cuts = pyo.ConstraintList()

    return model

# =====================================================================
# 2. BENDERS SUBPROBLEM
# =====================================================================
def build_benders_subproblem(bus_df, gen_df, branch_df, load_vector, baseMVA=100.0):
    """
    Builds the Feasibility Subproblem (Eq 25-28) for a single contingency.
    Validates if post-contingency flow respects thermal limits.
    """
    sub = pyo.ConcreteModel()
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    sub.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    sub.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    sub.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))

    bus_gens = {b: [] for b in sub.Buses}
    for _, row in gen_df.iterrows():
        bus_gens[row['bus_i']].append(row['gen_ID'])
        
    lines_from = {b: [] for b in sub.Buses}
    lines_to = {b: [] for b in sub.Buses}
    branch_ends = {}
    for _, row in branch_df.iterrows():
        lines_from[row['bus_i']].append(row['line_ID'])
        lines_to[row['bus_j']].append(row['line_ID'])
        branch_ends[row['line_ID']] = (row['bus_i'], row['bus_j'])

    # Variables
    sub.Pgs_tilde = pyo.Var(sub.Gens)
    sub.Theta_s = pyo.Var(sub.Buses, bounds=(-np.pi, np.pi))
    sub.Pf_s = pyo.Var(sub.Branches)
    
    # Slack variables for feasibility tracking 
    sub.v_plus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.v_minus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)

    # Mutable Parameter to receive the dispatch from the Master Problem
    sub.Pgs_star = pyo.Param(sub.Gens, mutable=True, initialize=0.0)

    # Objective: Minimize slack violations
    def sub_obj_rule(m):
        return sum(m.v_plus[b] + m.v_minus[b] for b in m.Buses)
    sub.feasibility_obj = pyo.Objective(rule=sub_obj_rule, sense=pyo.minimize)

    # Fix generation to Master Problem's solution (Extract duals here)
    def fix_g_rule(m, i):
        return m.Pgs_tilde[i] == m.Pgs_star[i]
    sub.fix_g = pyo.Constraint(sub.Gens, rule=fix_g_rule)

    def flow_s_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_s[l] == (m.Theta_s[bus_from] - m.Theta_s[bus_to]) / x[l]
    sub.flow_s_eq = pyo.Constraint(sub.Branches, rule=flow_s_rule)

    def limit_s_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return (-rateA[l], m.Pf_s[l], rateA[l])
    sub.limit_s_eq = pyo.Constraint(sub.Branches, rule=limit_s_rule)

    def balance_s_rule(m, b):
        gen_total = sum(m.Pgs_tilde[g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pf_s[l] for l in lines_from[b])
        flow_in = sum(m.Pf_s[l] for l in lines_to[b])
        return gen_total + m.v_plus[b] - m.v_minus[b] - load_total == flow_out - flow_in
    sub.balance_s_eq = pyo.Constraint(sub.Buses, rule=balance_s_rule)
    sub.ref_bus_s = pyo.Constraint(expr=sub.Theta_s[ref_bus_id] == 0)

    # Declare Suffix to extract Dual Variables (mu_s)
    sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return sub

# =====================================================================
# 3. ALGORITHM EXECUTION
# =====================================================================
def run_modified_benders(bus, gen, branch, gencost, load_vector, epsilon=1e-4):
    print("Building Master Problem (Warning: Exponential binary complexity)...")
    master = build_benders_master(bus, gen, branch, gencost, load_vector)
    subproblem = build_benders_subproblem(bus, gen, branch, load_vector)
    
    solver = pyo.SolverFactory('gurobi_direct')
    iteration = 1
    
    while True:
        print(f"--- Benders Iteration {iteration} ---")
        
        # 1. Solve Master Problem
        results = solver.solve(master, tee=False)
        if results.solver.termination_condition != TerminationCondition.optimal:
            raise ValueError("Master Problem became infeasible.")
            
        Pgs_star = {(s, i): pyo.value(master.Pgs[s, i]) for s in master.Contingencies for i in master.Gens}
        cuts_added = 0
        
        # 2. Solve Subproblems
        for s in master.Contingencies:
            for i in subproblem.Gens:
                subproblem.Pgs_star[i] = Pgs_star[(s, i)]
                
            sub_results = solver.solve(subproblem, tee=False)
            obj_val = pyo.value(subproblem.feasibility_obj)
            
            # 3. Add Feasibility Benders Cut
            if obj_val > epsilon:
                print(f"  -> Contingency {s} failed (Violation: {obj_val:.4f}). Adding Benders Cut.")
                mu = {i: subproblem.dual[subproblem.fix_g[i]] for i in subproblem.Gens}
                cut_expr = obj_val + sum(mu[i] * (master.Pgs[s, i] - Pgs_star[(s, i)]) for i in master.Gens) <= 0
                master.benders_cuts.add(cut_expr)
                cuts_added += 1
                
        if cuts_added == 0:
            print("Optimal and Secure Dispatch Found! No more cuts needed.")
            break
            
        iteration += 1

    optimal_g = {i: pyo.value(master.Pg[i]) * 100.0 for i in master.Gens}
    return optimal_g, iteration

# =====================================================================
# 4. MAIN DATA INGESTION & SAVING RESULTS
# =====================================================================
def calculate_generation_cost(pg_mw_dict, cost_df):
    """Calculates total generation cost using strictly linear coefficients."""
    cost = 0.0
    for i, pg_mw in pg_mw_dict.items():
        row = cost_df[cost_df['gen_ID'] == i].iloc[0]
        c1, c0 = row['c1'], row['c0']
        # Strictly linear cost calculation matching the paper
        cost += (c1 * pg_mw) + c0
    return cost

def main():
    parser = argparse.ArgumentParser(description="Run Modified Benders Decomposition")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case300_ieee")
    parser.add_argument('--num_tests', type=int, default=1, help="Number of scenarios to solve")
    args = parser.parse_args()

    # Load Base Data
    case_path = f'../excel_outputs/{args.case}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]

    case['gen']['Pmax'] = case['gen']['Pmax'].astype(float)
    case['gencost']['c1'] = case['gencost']['c1'].astype(float)
    case['gencost']['c2'] = case['gencost']['c2'].astype(float)
    
    # Pre-process base generator data
    case['gamma'] = 1 
    case['gen'].attrs['gamma'] = case['gamma'] 
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    bus_list = sorted(case['bus']['bus_i'].tolist())

    # Load the specific generated dataset
    csv_path = f'data/{args.case}_generated_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    dataset_df = pd.read_csv(csv_path)
    num_tests = min(args.num_tests, len(dataset_df))

    # Initialize a list to store the results
    benders_results = []

    for s in range(num_tests):
        print(f"\n=======================================================")
        print(f" SOLVING SCENARIO {s} VIA BENDERS DECOMPOSITION ")
        print(f"=======================================================")
        
        row = dataset_df.iloc[s]
        
        # Extract Perturbed Loads
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}
        
        # Inject Perturbed Limits and Costs into the DataFrames
        for gen_id in case['gen']['gen_ID']:
            case['gen'].loc[case['gen']['gen_ID'] == gen_id, 'Pmax'] = row[f"Gen_{gen_id}_Pmax"]
            case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c1'] = row[f"Gen_{gen_id}_c1"]
            case['gencost'].loc[case['gencost']['gen_ID'] == gen_id, 'c2'] = row[f"Gen_{gen_id}_c2"]

        start_time = time.time()
        
        # Execute Benders
        optimal_dispatch_mw, benders_iters = run_modified_benders(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector
        )
        
        solve_time = time.time() - start_time
        
        # Calculate objective cost
        total_cost = calculate_generation_cost(optimal_dispatch_mw, case['gencost'])
        
        print(f"\nScenario {s} finished in {solve_time:.2f} seconds.")
        print(f"Total Benders Iterations: {benders_iters}")
        print(f"Total Cost: ${total_cost:.2f}")

        # Store the metrics
        benders_results.append({
            'Scenario_ID': s,
            'Benders_Time_s': round(solve_time, 4),
            'Benders_Iterations': benders_iters,
            'Total_Cost_$': round(total_cost, 2)
        })

    # Save all results to a CSV file
    os.makedirs('data', exist_ok=True)
    out_filename = f"data/{args.case}_benders_results.csv"
    results_df = pd.DataFrame(benders_results)
    results_df.to_csv(out_filename, index=False)
    print(f"\n*** All Benders benchmarking results successfully saved to {out_filename} ***")

if __name__ == "__main__":
    main()