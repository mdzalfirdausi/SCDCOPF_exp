import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import numpy as np
import math

# ==========================================
# 1. MATRIX BUILDERS (PTDF & LODF)
# ==========================================

def build_ptdf(bus_df, branch_df, ref_bus_id):
    """Constructs the Power Transfer Distribution Factor (PTDF) matrix."""
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    
    bus_list = sorted(bus_df['bus_i'].tolist())
    num_buses = len(bus_list)
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    num_branches = len(branch_df)
    
    A = np.zeros((num_branches, num_buses))
    Bd = np.zeros((num_branches, num_branches))
    
    for idx, row in branch_df.iterrows():
        from_idx = bus_idx[row['bus_i']]
        to_idx = bus_idx[row['bus_j']]
        A[idx, from_idx] = 1.0
        A[idx, to_idx] = -1.0
        Bd[idx, idx] = 1.0 / row['x']
        
    B_bus = A.T @ Bd @ A
    ref_idx = bus_idx[ref_bus_id]
    non_ref_indices = [i for i in range(num_buses) if i != ref_idx]
    
    B_bus_reduced = B_bus[np.ix_(non_ref_indices, non_ref_indices)]
    X_bus_reduced = np.linalg.inv(B_bus_reduced)
    
    X_bus = np.zeros((num_buses, num_buses))
    X_bus[np.ix_(non_ref_indices, non_ref_indices)] = X_bus_reduced
    PTDF = Bd @ A @ X_bus
    
    return PTDF, bus_list

def build_lodf(PTDF_matrix, branch_df, bus_list):
    """Constructs the Line Outage Distribution Factor (LODF) matrix for N-1 Line Security."""
    num_branches = len(branch_df)
    LODF = np.zeros((num_branches, num_branches))
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    
    for k in range(num_branches):
        bus_i = bus_idx[branch_df.iloc[k]['bus_i']]
        bus_j = bus_idx[branch_df.iloc[k]['bus_j']]
        
        # Sensitivity of the outaged line to its own terminal injections
        denom = 1.0 - (PTDF_matrix[k, bus_i] - PTDF_matrix[k, bus_j])
        
        # Skip radial lines (denom ~ 0) to avoid division by zero (islanding)
        if abs(denom) < 1e-5:
            continue
            
        for l in range(num_branches):
            if l == k:
                LODF[l, k] = -1.0 # The outaged line drops to 0 flow
            else:
                LODF[l, k] = (PTDF_matrix[l, bus_i] - PTDF_matrix[l, bus_j]) / denom
                
    return LODF


# ==========================================
# 2. ALGEBRAIC CHECKERS (THE SUBPROBLEMS)
# ==========================================

def calculate_primary_response(optimal_g, contingency_s, gen_df, load_vector, baseMVA=100.0, tolerance=1e-5):
    total_demand = sum(load_vector.values()) / baseMVA
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    
    gamma_val = gen_df.attrs.get('gamma', 1.0) 
    gamma = {i: gamma_val for i in gen_df['gen_ID']} 
    
    n_low, n_high = 0.0, 1.0
    n_s = 0.5
    g_s = {}

    for _ in range(50):
        current_generation = 0.0
        for i in gen_df['gen_ID']:
            if i == contingency_s:
                g_s[i] = 0.0
            else:
                base_g = optimal_g[i] / baseMVA
                g_s[i] = min(base_g + n_s * gamma[i] * pmax[i], max(pmax[i], base_g))
            current_generation += g_s[i]
            
        mismatch = current_generation - total_demand
        if abs(mismatch) < tolerance: break
        elif mismatch < 0: n_low = n_s 
        else: n_high = n_s 
        n_s = (n_low + n_high) / 2.0
        
    return n_s, g_s

def check_contingency_violations(g_s, PTDF_matrix, load_vector, branch_df, bus_gen_map, baseMVA=100.0):
    bus_gen_totals = {b: 0.0 for b in load_vector.keys()}
    for gen_id, gen_val in g_s.items():
        bus_id = bus_gen_map[gen_id]
        bus_gen_totals[bus_id] += gen_val
        
    g_array = np.array([bus_gen_totals[b] for b in sorted(load_vector.keys())])
    d_array = np.array([load_vector[b]/baseMVA for b in sorted(load_vector.keys())])
    
    net_injections = g_array - d_array
    flows = PTDF_matrix.dot(net_injections)
    rateA = branch_df['rateA'].values / baseMVA
    
    max_violation = 0.0
    worst_line_idx = None
    
    for l, flow in enumerate(flows):
        limit = rateA[l]
        if limit > 0: 
            violation = abs(flow) - limit
            if violation > max_violation:
                max_violation = violation
                worst_line_idx = int(branch_df.iloc[l]['line_ID']) # Cast to int
                
    return max_violation, worst_line_idx

def check_line_violations(optimal_g, PTDF_matrix, LODF_matrix, load_vector, branch_df, bus_gen_map, baseMVA=100.0):
    """Checks for branch limit violations under N-1 LINE contingencies."""
    bus_gen_totals = {b: 0.0 for b in load_vector.keys()}
    for gen_id, gen_val in optimal_g.items(): # gen_val is in MW
        bus_id = bus_gen_map[gen_id]
        bus_gen_totals[bus_id] += (gen_val / baseMVA) # convert to PU
        
    g_array = np.array([bus_gen_totals[b] for b in sorted(load_vector.keys())])
    d_array = np.array([load_vector[b]/baseMVA for b in sorted(load_vector.keys())])
    
    # 1. Base Flows
    net_injections = g_array - d_array
    base_flows = PTDF_matrix.dot(net_injections)
    rateA = branch_df['rateA'].values / baseMVA
    num_branches = len(branch_df)
    
    max_violation = 0.0
    worst_outaged_line = None
    worst_violated_line = None
    
    # 2. Algebraic Post-Contingency Flows using LODF
    for k in range(num_branches):
        outaged_line_id = branch_df.iloc[k]['line_ID']
        
        if np.all(LODF_matrix[:, k] == 0):
            continue # Skip radial lines (LODF safely forced to 0)
            
        post_flows = base_flows + LODF_matrix[:, k] * base_flows[k]
        
        for l in range(num_branches):
            if l == k: continue 
            limit = rateA[l]
            if limit > 0:
                violation = abs(post_flows[l]) - limit
                if violation > max_violation:
                    max_violation = violation
                    worst_outaged_line = int(outaged_line_id) # Cast to int
                    worst_violated_line = int(branch_df.iloc[l]['line_ID']) # Cast to int
                    
    return max_violation, worst_outaged_line, worst_violated_line


# ==========================================
# 3. THE CCGA EXACT MASTER PROBLEM
# ==========================================

def build_and_solve_ccga_master(bus_df, gen_df, branch_df, cost_df, load_vector, active_Kg, active_Ke, baseMVA=100.0, time_limit=None, log_file=None):
    """Builds the CCGA Master Problem for both Generator and Line Contingencies."""
    model = pyo.ConcreteModel()
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    model.All_Contingencies = pyo.Set(initialize=model.Gens)
    
    # Segregated Active Contingency Sets
    model.Active_Kg = pyo.Set(initialize=active_Kg) 
    model.Active_Ke = pyo.Set(initialize=active_Ke) 

    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    c1 = dict(zip(cost_df['gen_ID'], cost_df['c1']))
    c2 = dict(zip(cost_df['gen_ID'], cost_df['c2']))
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    
    gamma_val = gen_df.attrs.get('gamma', 1.0)
    gamma = {i: gamma_val for i in model.Gens} 

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

    # --- NOMINAL STATE (Base Case) ---
    model.Pg = pyo.Var(model.Gens) 
    model.Theta = pyo.Var(model.Buses, bounds=(-math.pi, math.pi))
    model.Pf = pyo.Var(model.Branches)
    
    model.Sl_nom_line = pyo.Var(model.Branches, within=pyo.NonNegativeReals)
    model.Sl_Pg_nom_up = pyo.Var(model.Gens, within=pyo.NonNegativeReals)
    model.Sl_Pg_nom_down = pyo.Var(model.Gens, within=pyo.NonNegativeReals)

    # --- PROVISIONAL VARIABLES (All Generator Scenarios) ---
    model.Pgs_prov = pyo.Var(model.All_Contingencies, model.Gens)
    model.Sl_Pg_prov_up = pyo.Var(model.All_Contingencies, model.Gens, within=pyo.NonNegativeReals)
    model.Sl_Pg_prov_down = pyo.Var(model.All_Contingencies, model.Gens, within=pyo.NonNegativeReals)

    # --- ACTIVE GENERATOR CONTINGENCIES (Kg) ---
    if len(active_Kg) > 0:
        model.ns_g = pyo.Var(model.Active_Kg, bounds=(0, 1))
        model.xs_g = pyo.Var(model.Active_Kg, model.Gens, domain=pyo.Binary)
        model.Thetas_g = pyo.Var(model.Active_Kg, model.Buses, bounds=(-math.pi, math.pi))
        model.Pfs_g = pyo.Var(model.Active_Kg, model.Branches)
        model.Sl_cont_line_g = pyo.Var(model.Active_Kg, model.Branches, within=pyo.NonNegativeReals)

    # --- ACTIVE LINE CONTINGENCIES (Ke) ---
    if len(active_Ke) > 0:
        model.Thetas_e = pyo.Var(model.Active_Ke, model.Buses, bounds=(-math.pi, math.pi))
        model.Pfs_e = pyo.Var(model.Active_Ke, model.Branches)
        model.Sl_cont_line_e = pyo.Var(model.Active_Ke, model.Branches, within=pyo.NonNegativeReals)

    # --- OBJECTIVE FUNCTION ---
    def obj_rule(m):
        penalty = 1500.0  # From the 2025 paper
        
        gen_cost = sum(c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
        
        slack_cost_nom = penalty * (
            sum(m.Sl_nom_line[l] for l in m.Branches) + 
            sum(m.Sl_Pg_nom_up[i] + m.Sl_Pg_nom_down[i] for i in m.Gens)
        )
        
        slack_cost_prov = penalty * sum(m.Sl_Pg_prov_up[s, i] + m.Sl_Pg_prov_down[s, i] for s in m.All_Contingencies for i in m.Gens)
        
        slack_cost_cont_g = penalty * sum(m.Sl_cont_line_g[s, l] for s in m.Active_Kg for l in m.Branches) if len(active_Kg) > 0 else 0.0
        slack_cost_cont_e = penalty * sum(m.Sl_cont_line_e[e, l] for e in m.Active_Ke for l in m.Branches) if len(active_Ke) > 0 else 0.0
            
        return gen_cost + slack_cost_nom + slack_cost_prov + slack_cost_cont_g + slack_cost_cont_e
        
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- NOMINAL CONSTRAINTS ---
    model.flow_eq = pyo.Constraint(model.Branches, rule=lambda m, l: m.Pf[l] == (m.Theta[branch_ends[l][0]] - m.Theta[branch_ends[l][1]]) / x[l])
    model.limit_upper_eq = pyo.Constraint(model.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pf[l] - m.Sl_nom_line[l] <= rateA[l])
    model.limit_lower_eq = pyo.Constraint(model.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pf[l] + m.Sl_nom_line[l] >= -rateA[l])
    model.pg_upper_eq = pyo.Constraint(model.Gens, rule=lambda m, i: m.Pg[i] - m.Sl_Pg_nom_up[i] <= pmax[i])
    model.pg_lower_eq = pyo.Constraint(model.Gens, rule=lambda m, i: m.Pg[i] + m.Sl_Pg_nom_down[i] >= pmin[i])
    
    def balance_rule(m, b):
        return sum(m.Pg[g] for g in bus_gens[b]) - (load_vector[b] / baseMVA) == sum(m.Pf[l] for l in lines_from[b]) - sum(m.Pf[l] for l in lines_to[b])
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)
    model.ref_bus = pyo.Constraint(expr=model.Theta[ref_bus_id] == 0)

    # --- PROVISIONAL CONSTRAINTS ---
    model.failed_gen_prov_eq = pyo.Constraint(model.All_Contingencies, rule=lambda m, s: m.Pgs_prov[s, s] == 0)
    model.global_demand_prov_eq = pyo.Constraint(model.All_Contingencies, rule=lambda m, s: sum(m.Pgs_prov[s, i] for i in m.Gens) == sum(load_vector[b] for b in m.Buses) / baseMVA)
    model.provisional_bound_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] - m.Pg[i] <= gamma[i] * pmax[i])
    model.pgs_prov_upper_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] - m.Sl_Pg_prov_up[s, i] <= pmax[i])
    model.pgs_prov_lower_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] + m.Sl_Pg_prov_down[s, i] >= 0)

    # --- GENERATOR CONTINGENCY BLOCK (Kg) ---
    if len(active_Kg) > 0:
        model.apr_upper_eq = pyo.Constraint(model.Active_Kg, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] - m.Pg[i] - m.ns_g[s] * gamma[i] * pmax[i] <= pmax[i] * (1 - m.xs_g[s, i]))
        model.apr_lower_eq = pyo.Constraint(model.Active_Kg, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] - m.Pg[i] - m.ns_g[s] * gamma[i] * pmax[i] >= -pmax[i] * (1 - m.xs_g[s, i]))
        model.apr_limit_eq1 = pyo.Constraint(model.Active_Kg, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pg[i] + m.ns_g[s] * gamma[i] * pmax[i] >= pmax[i] * (1 - m.xs_g[s, i]))
        model.apr_limit_eq2 = pyo.Constraint(model.Active_Kg, model.Gens, rule=lambda m, s, i: pyo.Constraint.Skip if s == i else m.Pgs_prov[s, i] >= pmax[i] * (1 - m.xs_g[s, i]))
        model.flow_cont_g_eq = pyo.Constraint(model.Active_Kg, model.Branches, rule=lambda m, s, l: m.Pfs_g[s, l] == (m.Thetas_g[s, branch_ends[l][0]] - m.Thetas_g[s, branch_ends[l][1]]) / x[l])
        model.limit_cont_upper_g_eq = pyo.Constraint(model.Active_Kg, model.Branches, rule=lambda m, s, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pfs_g[s, l] - m.Sl_cont_line_g[s, l] <= rateA[l])
        model.limit_cont_lower_g_eq = pyo.Constraint(model.Active_Kg, model.Branches, rule=lambda m, s, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pfs_g[s, l] + m.Sl_cont_line_g[s, l] >= -rateA[l])
        
        def balance_cont_g_rule(m, s, b):
            return sum(m.Pgs_prov[s, g] for g in bus_gens[b]) - (load_vector[b] / baseMVA) == sum(m.Pfs_g[s, l] for l in lines_from[b]) - sum(m.Pfs_g[s, l] for l in lines_to[b])
        model.balance_cont_g_eq = pyo.Constraint(model.Active_Kg, model.Buses, rule=balance_cont_g_rule)
        model.ref_bus_cont_g_eq = pyo.Constraint(model.Active_Kg, rule=lambda m, s: m.Thetas_g[s, ref_bus_id] == 0)

    # --- LINE CONTINGENCY BLOCK (Ke) ---
    if len(active_Ke) > 0:
        def flow_cont_e_rule(m, e, l):
            if l == e: return m.Pfs_e[e, l] == 0 # Outaged line forced to zero flow
            return m.Pfs_e[e, l] == (m.Thetas_e[e, branch_ends[l][0]] - m.Thetas_e[e, branch_ends[l][1]]) / x[l]
        model.flow_cont_e_eq = pyo.Constraint(model.Active_Ke, model.Branches, rule=flow_cont_e_rule)
        
        model.limit_cont_upper_e_eq = pyo.Constraint(model.Active_Ke, model.Branches, rule=lambda m, e, l: pyo.Constraint.Skip if (rateA[l] == 0 or l == e) else m.Pfs_e[e, l] - m.Sl_cont_line_e[e, l] <= rateA[l])
        model.limit_cont_lower_e_eq = pyo.Constraint(model.Active_Ke, model.Branches, rule=lambda m, e, l: pyo.Constraint.Skip if (rateA[l] == 0 or l == e) else m.Pfs_e[e, l] + m.Sl_cont_line_e[e, l] >= -rateA[l])
        
        def balance_cont_e_rule(m, e, b):
            # Gen total remains identical to Base Case Pg (generators do not trip)
            return sum(m.Pg[g] for g in bus_gens[b]) - (load_vector[b] / baseMVA) == sum(m.Pfs_e[e, l] for l in lines_from[b]) - sum(m.Pfs_e[e, l] for l in lines_to[b])
        model.balance_cont_e_eq = pyo.Constraint(model.Active_Ke, model.Buses, rule=balance_cont_e_rule)
        model.ref_bus_cont_e_eq = pyo.Constraint(model.Active_Ke, rule=lambda m, e: m.Thetas_e[e, ref_bus_id] == 0)

    # --- SOLVE ---
    solver = pyo.SolverFactory('gurobi')
    if time_limit:
        solver.options['TimeLimit'] = 60 # Give it plenty of time to read the file
        solver.options['NodeLimit'] = 0  # FORCE it to stop immediately after Presolve!
        
    # Pyomo handles log files natively via the solve() method, not solver options!
    results = solver.solve(model, tee=(log_file is not None), logfile=log_file)
    
    if results.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        if time_limit: 
            return {}, TerminationCondition.maxTimeLimit
        raise ValueError(f"Model mathematically infeasible. Check grid data format.")
        
    # If we hit the limit, there won't be a solution to load, which is fine for the log test.
    if time_limit and len(model.Pg) > 0 and pyo.value(model.Pg[model.Gens.first()], exception=False) is None:
        return {}, TerminationCondition.maxTimeLimit

    optimal_g = {i: pyo.value(model.Pg[i]) * baseMVA for i in model.Gens}
    return optimal_g, results.solver.termination_condition


# ==========================================
# 4. MASTER-SUBPROBLEM ALGORITHM LOOP
# ==========================================

def run_ccga_algorithm(bus, gen, branch, gencost, load_vector, PTDF_matrix, initial_active_Kg=None, initial_active_Ke=None):
    """Executes the CCGA Loop dynamically addressing Gen (Kg) and Line (Ke) outages."""
    active_Kg = initial_active_Kg.copy() if initial_active_Kg is not None else []
    active_Ke = initial_active_Ke.copy() if initial_active_Ke is not None else []
    
    epsilon = 0.05 / 100.0 # Tolerance in per-unit
    iteration = 0
    bus_gen_map = gen.set_index('gen_ID')['bus_i'].to_dict()
    
    # Pre-calculate LODF for rapid line subproblem checking
    bus_list = sorted(bus['bus_i'].tolist())
    LODF_matrix = build_lodf(PTDF_matrix, branch, bus_list)
    
    while True:
        optimal_g, status = build_and_solve_ccga_master(
            bus, gen, branch, gencost, load_vector, active_Kg, active_Ke
        )
        
        # --- Check N-1 Generator Violations ---
        global_max_viol_gen = 0.0
        worst_gen_cont = None
        worst_line_under_gen = None
        
        for s in gen['gen_ID']:
            n_s, g_s = calculate_primary_response(optimal_g, s, gen, load_vector)
            max_viol, line_idx = check_contingency_violations(g_s, PTDF_matrix, load_vector, branch, bus_gen_map)
            
            if max_viol > global_max_viol_gen:
                global_max_viol_gen = max_viol
                worst_gen_cont = s
                worst_line_under_gen = line_idx
                
        # --- Check N-1 Line Violations ---
        global_max_viol_line, worst_line_cont, worst_line_under_line = check_line_violations(
            optimal_g, PTDF_matrix, LODF_matrix, load_vector, branch, bus_gen_map
        )

        # --- Convergence and Update Logic ---
        if global_max_viol_gen <= epsilon and global_max_viol_line <= epsilon:
            print("      -> Network is secure against all N-1 Gen and Line contingencies.")
            break
            
        # Target the worst overall physical violation
        if global_max_viol_gen > global_max_viol_line:
            if worst_gen_cont not in active_Kg:
                active_Kg.append(worst_gen_cont)
                print(f"      -> Iteration {iteration}: Line {worst_line_under_gen} violated under loss of GEN {worst_gen_cont}. Adding to Master.")
            else:
                print(f"      -> Iteration {iteration}: GEN {worst_gen_cont} cannot be physically secured. Slack variable active. Ending loop.")
                break 
        else:
            if worst_line_cont not in active_Ke:
                active_Ke.append(worst_line_cont)
                print(f"      -> Iteration {iteration}: Line {worst_line_under_line} violated under loss of LINE {worst_line_cont}. Adding to Master.")
            else:
                print(f"      -> Iteration {iteration}: LINE {worst_line_cont} cannot be physically secured. Slack variable active. Ending loop.")
                break 
                
        iteration += 1
        
    return optimal_g, status, iteration, active_Kg, active_Ke

# ==========================================
# 5. EXTENSIVE SCOPF STATS GENERATOR
# ==========================================
if __name__ == "__main__":
    import argparse
    import os
    import re
    
    parser = argparse.ArgumentParser(description="Extract Grid Topology and MILP Stats")
    parser.add_argument('--case', type=str, default="pglib_opf_case300_ieee")
    args = parser.parse_args()
    
    case_name = args.case
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    if not os.path.exists(case_path):
        print(f"Error: Could not find {case_path}")
        exit()
        
    print(f"Loading {case_name} to extract network statistics...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','branch', 'gencost'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    bus_df = case['bus']
    gen_df = case['gen']
    branch_df = case['branch']

    # # 1. Topological Stats
    # N = len(bus_df)
    # G = len(gen_df)
    # E = len(branch_df)
    
    # # Loads (|L|) - Count of buses with active demand > 0
    # L = len(bus_df[bus_df['Pd'] != 0]) if 'Pd' in bus_df.columns else 0
    
    # # Filter generators for Kg (Drop 0 capacity)
    # zero_gen_idx = [num for num, i in enumerate(gen_df.Pmax.values / baseMVA) 
    #                 if (i == 0 and (gen_df.Pmin.values / baseMVA)[num] == 0) or 
    #                 (gen_df.Pmin.values / baseMVA)[num] < 0]
    
    # Kg = G - len(zero_gen_idx)
    
    # # Build PTDF to check Line Contingencies (Ke)
    # ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    # PTDF_matrix, bus_list = build_ptdf(bus_df, branch_df, ref_bus_id)
    
    # # Count Ke (Skipping radial lines just like build_lodf)
    # Ke = 0
    # bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    # for k in range(E):
    #     bus_i = bus_idx[branch_df.iloc[k]['bus_i']]
    #     bus_j = bus_idx[branch_df.iloc[k]['bus_j']]
    #     denom = 1.0 - (PTDF_matrix[k, bus_i] - PTDF_matrix[k, bus_j])
    #     if abs(denom) >= 1e-5:  # Not radial
    #         Ke += 1
            
    # # 3. Print Results
    # print("\n" + "="*80)
    # print(" TOPOLOGICAL NETWORK STATISTICS (For the Image Table)")
    # print("="*80)
    # print(f"{'Test Case':<15} | {'|N|':<5} | {'|G|':<5} | {'|L|':<5} | {'|E|':<5} | {'|Kg|':<5} | {'|Ke|':<5}")
    # print("-" * 65)
    # print(f"{case_name:<15} | {N:<5} | {G:<5} | {L:<5} | {E:<5} | {Kg:<5} | {Ke:<5}")
    
    # Filter active generators for Kg
    zero_gen_idx = [num for num, i in enumerate(gen_df.Pmax.values / baseMVA) 
                    if (i == 0 and (gen_df.Pmin.values / baseMVA)[num] == 0) or 
                    (gen_df.Pmin.values / baseMVA)[num] < 0]
    active_gen_df = gen_df.drop(index=zero_gen_idx)
    
    # Build PTDF to find active Line Contingencies (Ke)
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    PTDF_matrix, bus_list = build_ptdf(bus_df, branch_df, ref_bus_id)
    
    active_Ke_list = []
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    for k in range(len(branch_df)):
        bus_i = bus_idx[branch_df.iloc[k]['bus_i']]
        bus_j = bus_idx[branch_df.iloc[k]['bus_j']]
        denom = 1.0 - (PTDF_matrix[k, bus_i] - PTDF_matrix[k, bus_j])
        if abs(denom) >= 1e-5:  # Not radial
            active_Ke_list.append(int(branch_df.iloc[k]['line_ID']))
            
    active_Kg_list = active_gen_df['gen_ID'].tolist()
    
    # Create a generic base load vector
    load_vector = {b: bus_df.loc[bus_df['bus_i'] == b, 'Pd'].values[0] for b in bus_list}
    
    print(f"\nBuilding Extensive SCOPF Pyomo Model (|Kg|={len(active_Kg_list)}, |Ke|={len(active_Ke_list)})...")
    print("Handing to Gurobi for Presolve analysis (Time Limit: 1s)...")
    
    log_filename = f"gurobi_presolve_{case_name}.log"
    
    try:
        build_and_solve_ccga_master(
            bus_df, active_gen_df, branch_df, case['gencost'], load_vector, 
            active_Kg_list, active_Ke_list, baseMVA, 
            time_limit=1, log_file=log_filename
        )
    except Exception as e:
        print(f"\n[Note] Solver execution interrupted (Expected if hitting 1s TimeLimit).")
        print(f"Message: {e}\n")

    # Parse the Gurobi Log File
    with open(log_filename, 'r') as f:
        log_text = f.read()

    # Regex to find Before Presolve Stats
    rows_match = re.search(r'Optimize a model with (\d+) rows', log_text)
    vars_match = re.search(r'Variable types: (\d+) continuous, \d+ integer \((\d+) binary\)', log_text)
    
    # Regex to find After Presolve Stats
    presolved_match = re.search(r'Presolved: (\d+) rows, (\d+) columns', log_text)
    
    if rows_match and vars_match and presolved_match:
        before_cnst = int(rows_match.group(1))
        before_cv = int(vars_match.group(1))
        bv = int(vars_match.group(2)) # Binary variables usually don't get presolved away in this problem
        
        after_cnst = int(presolved_match.group(1))
        after_total_vars = int(presolved_match.group(2))
        after_cv = after_total_vars - bv
        
        print("\n" + "="*80)
        print(" TABLE II: EXTENSIVE SCOPF STATISTICS (Before and After Presolve)")
        print("="*80)
        print(f"{'':<15} | {'--- BEFORE PRESOLVE ---':<28} | {'--- AFTER PRESOLVE ---'}")
        print(f"{'Test Case':<15} | {'#CV':<8} | {'#BV':<8} | {'#Cnst':<8} | {'#CV':<8} | {'#BV':<8} | {'#Cnst':<8}")
        print("-" * 80)
        
        # Format like the paper (e.g., 44.5k)
        print(f"{case_name:<15} | {before_cv/1000:>6.1f}k | {bv/1000:>6.1f}k | {before_cnst/1000:>6.1f}k "
              f"| {after_cv/1000:>6.1f}k | {bv/1000:>6.1f}k | {after_cnst/1000:>6.1f}k")
        print("="*80 + "\n")
    else:
        print("Error: Could not parse Gurobi log file. Please check gurobi_presolve.log")
        
    # Clean up log file
    if os.path.exists(log_filename):
        os.remove(log_filename)