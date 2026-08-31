import os
import time
import argparse
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

# =====================================================================
# 1. THE ADMM MASTER PROBLEM (Nominal Dispatch + AL Cuts)
# =====================================================================
def build_admm_master(bus_df, gen_df, branch_df, cost_df, load_vector, baseMVA=100.0):
    """
    Builds the Master Problem for the nominal dispatch. 
    It will receive AL cuts (which include binary variables) iteratively.
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
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))

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

    # --- Master Variables ---
    model.Pg = pyo.Var(model.Gens) 
    model.Theta = pyo.Var(model.Buses, bounds=(-np.pi, np.pi))
    model.Pf = pyo.Var(model.Branches)
    
    # Cost-to-go estimators for each contingency
    model.t_k = pyo.Var(model.Contingencies, within=pyo.NonNegativeReals)

    # --- Objective ---
    def obj_rule(m):
        nominal_cost = sum(c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
        future_cost = sum(m.t_k[k] for k in m.Contingencies)
        return nominal_cost + future_cost
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Nominal Constraints ---
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

    # Dictionary to store maximum capacities for Big-M bounds in AL Cuts
    model.pmax_dict = pmax 
    return model

def add_al_cut_to_master(master, iter_idx, k, P_val, g_bar, mu, beta):
    """
    Adds the AL Cut: t_k >= P - <mu, g - g_bar> - beta * ||g - g_bar||_1.
    Uses the exact MILP formulation for -L1 norm.
    """
    b = pyo.Block()
    b.Gens = pyo.Set(initialize=master.Gens)
    
    # Auxiliary variables to model the negative absolute value
    b.z = pyo.Var(b.Gens, domain=pyo.Binary)
    b.yp = pyo.Var(b.Gens, within=pyo.NonNegativeReals)
    b.ym = pyo.Var(b.Gens, within=pyo.NonNegativeReals)

    def diff_rule(b, i):
        return b.yp[i] - b.ym[i] == master.Pg[i] - g_bar[i]
    b.diff_eq = pyo.Constraint(b.Gens, rule=diff_rule)

    # Big-M bounds
    def bigm_p_rule(b, i):
        return b.yp[i] <= master.pmax_dict[i] * b.z[i]
    b.bigm_p_eq = pyo.Constraint(b.Gens, rule=bigm_p_rule)

    def bigm_m_rule(b, i):
        return b.ym[i] <= master.pmax_dict[i] * (1 - b.z[i])
    b.bigm_m_eq = pyo.Constraint(b.Gens, rule=bigm_m_rule)

    # The actual AL Cut Constraint
    def cut_rule(b):
        return master.t_k[k] >= P_val - sum(mu[i] * (master.Pg[i] - g_bar[i]) for i in b.Gens) \
                                      - beta * sum(b.yp[i] + b.ym[i] for i in b.Gens)
    b.cut_eq = pyo.Constraint(rule=cut_rule)

    master.add_component(f"AL_Cut_Iter{iter_idx}_Cont{k}", b)


# =====================================================================
# 2. THE ADMM SUBPROBLEM (Contingency Evaluation)
# =====================================================================
def solve_contingency_subproblem(k_cont, bus_df, gen_df, branch_df, load_vector, g_bar, mu, beta, baseMVA=100.0):
    """
    Solves the local L1-penalized subproblem for a single contingency.
    """
    sub = pyo.ConcreteModel()
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    sub.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    sub.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    sub.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    
    gamma = {i: gen_df.attrs.get('gamma', 0.05) for i in sub.Gens}
    g_hat = {i: pmax[i] - pmin[i] for i in sub.Gens}

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
    sub.Pg_prov = pyo.Var(sub.Gens, within=pyo.NonNegativeReals) # Local copy of base dispatch
    sub.Pgs = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)     # Post-contingency dispatch
    sub.Theta_s = pyo.Var(sub.Buses, bounds=(-np.pi, np.pi))
    sub.Pf_s = pyo.Var(sub.Branches)
    
    sub.n_s = pyo.Var(bounds=(0, 1))
    sub.x_s = pyo.Var(sub.Gens, domain=pyo.Binary)

    # Absolute value modeling for + ||g_prov - g_bar||_1
    sub.d_plus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    sub.d_minus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    
    def abs_diff_rule(m, i):
        return m.d_plus[i] - m.d_minus[i] == m.Pg_prov[i] - g_bar[i]
    sub.abs_diff_eq = pyo.Constraint(sub.Gens, rule=abs_diff_rule)

    # Soft constraints for feasibility tracking
    sub.v_plus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.v_minus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.sl_line = pyo.Var(sub.Branches, within=pyo.NonNegativeReals)

    # Objective: Contingency Penalty + AL terms
    M_eta = 1500.0
    def sub_obj_rule(m):
        penalty = M_eta * sum(m.v_plus[b] + m.v_minus[b] for b in m.Buses) + \
                  M_eta * sum(m.sl_line[l] for l in m.Branches)
        al_term = sum(mu[i] * (m.Pg_prov[i] - g_bar[i]) + beta * (m.d_plus[i] + m.d_minus[i]) for i in m.Gens)
        return penalty + al_term
    sub.obj = pyo.Objective(rule=sub_obj_rule, sense=pyo.minimize)

    # --- Contingency Constraints ---
    sub.failed_gen_eq = pyo.Constraint(expr=sub.Pgs[k_cont] == 0)

    def global_demand_rule(m):
        total_load = sum(load_vector[b] / baseMVA for b in m.Buses)
        return sum(m.Pgs[i] for i in m.Gens) == total_load
    sub.global_demand_eq = pyo.Constraint(rule=global_demand_rule)

    # Primary Response (APR) using local provisional dispatch
    def apr_rule1(m, i):
        if i == k_cont: return pyo.Constraint.Skip
        return m.Pgs[i] - m.Pg_prov[i] - m.n_s * gamma[i] * g_hat[i] <= pmax[i] * (1 - m.x_s[i])
    sub.apr_eq1 = pyo.Constraint(sub.Gens, rule=apr_rule1)

    def apr_rule2(m, i):
        if i == k_cont: return pyo.Constraint.Skip
        return m.Pgs[i] - m.Pg_prov[i] - m.n_s * gamma[i] * g_hat[i] >= -pmax[i] * (1 - m.x_s[i])
    sub.apr_eq2 = pyo.Constraint(sub.Gens, rule=apr_rule2)

    def apr_rule3(m, i):
        if i == k_cont: return pyo.Constraint.Skip
        return m.Pg_prov[i] + m.n_s * gamma[i] * g_hat[i] >= pmax[i] * (1 - m.x_s[i])
    sub.apr_eq3 = pyo.Constraint(sub.Gens, rule=apr_rule3)

    def apr_rule4(m, i):
        if i == k_cont: return pyo.Constraint.Skip
        return m.Pgs[i] >= pmax[i] * (1 - m.x_s[i])
    sub.apr_eq4 = pyo.Constraint(sub.Gens, rule=apr_rule4)

    def flow_s_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_s[l] == (m.Theta_s[bus_from] - m.Theta_s[bus_to]) / x[l]
    sub.flow_s_eq = pyo.Constraint(sub.Branches, rule=flow_s_rule)

    def limit_s_upper_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pf_s[l] - m.sl_line[l] <= rateA[l]
    sub.limit_s_upper_eq = pyo.Constraint(sub.Branches, rule=limit_s_upper_rule)

    def limit_s_lower_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pf_s[l] + m.sl_line[l] >= -rateA[l]
    sub.limit_s_lower_eq = pyo.Constraint(sub.Branches, rule=limit_s_lower_rule)

    def balance_s_rule(m, b):
        gen_total = sum(m.Pgs[g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pf_s[l] for l in lines_from[b])
        flow_in = sum(m.Pf_s[l] for l in lines_to[b])
        return gen_total + m.v_plus[b] - m.v_minus[b] - load_total == flow_out - flow_in
    sub.balance_s_eq = pyo.Constraint(sub.Buses, rule=balance_s_rule)
    sub.ref_bus_s = pyo.Constraint(expr=sub.Theta_s[ref_bus_id] == 0)

    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['Threads'] = 128
    results = solver.solve(sub, tee=False)
    
    P_val = pyo.value(sub.obj)
    local_g_prov = {i: pyo.value(sub.Pg_prov[i]) for i in sub.Gens}
    
    return P_val, local_g_prov

# =====================================================================
# 3. ALGORITHM 4.1 EXECUTION
# =====================================================================
def run_admm_al_cuts(bus, gen, branch, gencost, load_vector, max_iters=50, tolerance=1e-2):
    print("Building ADMM Master Problem...")
    master = build_admm_master(bus, gen, branch, gencost, load_vector)
    solver = pyo.SolverFactory('gurobi_direct')
    
    gens_list = gen['gen_ID'].tolist()
    
    # Initialization
    g_bar = {i: 0.0 for i in gens_list}
    mu = {k: {i: 0.0 for i in gens_list} for k in gens_list}
    beta = {k: 5.0 for k in gens_list}  
    admmDualStepSize = 0.5
    
    prev_obj_val = 0.0
    final_iteration = 0

    for iteration in range(1, max_iters + 1):
        final_iteration = iteration
        print(f"\n--- ADMM Iteration {iteration} ---")
        
        # 1. Parallel Subproblems (x-update) & Cut Generation
        for k in gens_list:
            P_val, local_g_prov = solve_contingency_subproblem(
                k, bus, gen, branch, load_vector, g_bar, mu[k], beta[k]
            )
            add_al_cut_to_master(master, iteration, k, P_val, g_bar.copy(), mu[k], beta[k])
            
            # Dual Update 
            for i in gens_list:
                mu[k][i] += admmDualStepSize * beta[k] * (local_g_prov[i] - g_bar[i])
            beta[k] = min(100.0, beta[k] * 1.1) 
        
        # 2. Master Problem Solve (z-update)
        print(f"Solving Master Problem with {iteration * len(gens_list)} active AL Cuts...")
        results = solver.solve(master, tee=False)
        
        if results.solver.termination_condition != TerminationCondition.optimal:
            print("Master problem failed or unbounded.")
            break
            
        g_bar = {i: pyo.value(master.Pg[i]) for i in gens_list}
        obj_val = pyo.value(master.cost)
        print(f"Master Objective: ${obj_val:.2f}")

        # 3. Dynamic Stopping Check
        if abs(obj_val - prev_obj_val) < tolerance and iteration > 1:
            print("Optimal AL Cut consensus reached! Objective stagnated.")
            break
            
        prev_obj_val = obj_val

    optimal_g = {i: g_bar[i] * 100.0 for i in gens_list}
    return optimal_g, prev_obj_val, final_iteration

# =====================================================================
# 4. MAIN DATA INGESTION & SAVING
# =====================================================================
def calculate_true_cost(pg_mw_dict, cost_df):
    """Calculates the actual quadratic generation cost of the ADMM dispatch."""
    cost = 0.0
    for i, pg_mw in pg_mw_dict.items():
        row = cost_df[cost_df['gen_ID'] == i].iloc[0]
        c2, c1, c0 = row['c2'], row['c1'], row['c0']
        cost += (c2 * (pg_mw**2)) + (c1 * pg_mw) + c0
    return cost

def main():
    parser = argparse.ArgumentParser(description="Run ADMM with AL Cuts")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case118_ieee")
    parser.add_argument('--num_tests', type=int, default=1, help="Number of scenarios to solve")
    args = parser.parse_args()

    case_path = f'../excel_outputs/{args.case}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]

    # Cast to float to prevent Pandas LossySetitemError
    case['gen']['Pmax'] = case['gen']['Pmax'].astype(float)
    case['gencost']['c1'] = case['gencost']['c1'].astype(float)
    case['gencost']['c2'] = case['gencost']['c2'].astype(float)
    
    case['gamma'] = 1 
    case['gen'].attrs['gamma'] = case['gamma'] 
    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    bus_list = sorted(case['bus']['bus_i'].tolist())

    csv_path = f'data/{args.case}_generated_loads.csv'
    dataset_df = pd.read_csv(csv_path)
    num_tests = min(args.num_tests, len(dataset_df))

    admm_results = []

    for s in range(num_tests):
        print(f"\n=======================================================")
        print(f" SOLVING SCENARIO {s} VIA ADMM WITH AL CUTS ")
        print(f"=======================================================")
        
        row = dataset_df.iloc[s]
        load_vector = {b: row[f"Bus_{b}_Pd"] for b in bus_list}

        start_time = time.time()
        
        # Increased max_iters to 50 so it can naturally find consensus
        optimal_dispatch_mw, final_lb, total_iters = run_admm_al_cuts(
            case['bus'], case['gen'], case['branch'], case['gencost'], load_vector, max_iters=50
        )
        
        solve_time = time.time() - start_time
        actual_cost = calculate_true_cost(optimal_dispatch_mw, case['gencost'])
        
        print(f"\nScenario {s} finished in {solve_time:.2f} seconds.")
        print(f"Total Iterations: {total_iters}")
        print(f"ADMM Lower Bound: ${final_lb:.2f}")
        print(f"Actual Grid Cost: ${actual_cost:.2f}")
        
        admm_results.append({
            'Scenario': s,
            'ADMM_Time_s': round(solve_time, 4),
            'ADMM_Iterations': total_iters,
            'Master_LB_Cost_$': round(final_lb, 2),
            'Actual_Cost_$': round(actual_cost, 2)
        })

    os.makedirs('data/admm_al', exist_ok=True)
    out_filename = f"data/admm_al/{args.case}_admm_al_results.csv"
    pd.DataFrame(admm_results).to_csv(out_filename, index=False)
    print(f"\n*** Results saved to {out_filename} ***")

if __name__ == "__main__":
    main()