import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import numpy as np
import math

def build_ptdf(bus_df, branch_df, ref_bus_id):
    """Constructs the Power Transfer Distribution Factor (PTDF) matrix."""
    
    # Automatically find the reference bus (bus_type == 3)
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

def calculate_primary_response(optimal_g, contingency_s, gen_df, load_vector, baseMVA=100.0, tolerance=1e-5):
    """Algorithm 1: Bisection search to find the global signal n_s and exact g_s."""
    total_demand = sum(load_vector.values()) / baseMVA
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    
    # Extract gamma dynamically, defaulting to 1.0 if not found
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
                # If the exact solver used a slack variable, base_g might exceed pmax.
                # This max() ensures the bisection starts from the solver's physical reality.
                base_g = optimal_g[i] / baseMVA
                g_s[i] = min(base_g + n_s * gamma[i] * pmax[i], max(pmax[i], base_g))
            current_generation += g_s[i]
            
        mismatch = current_generation - total_demand
        if abs(mismatch) < tolerance:
            break
        elif mismatch < 0:
            n_low = n_s 
        else:
            n_high = n_s 
        n_s = (n_low + n_high) / 2.0
        
    return n_s, g_s

def check_contingency_violations(g_s, PTDF_matrix, load_vector, branch_df, bus_gen_map, baseMVA=100.0):
    """Algebraically calculates post-contingency line flows and finds maximum violations."""
    
    # 1. Map generation properly to physical buses using the FAST dictionary
    bus_gen_totals = {b: 0.0 for b in load_vector.keys()}
    for gen_id, gen_val in g_s.items():
        bus_id = bus_gen_map[gen_id]
        bus_gen_totals[bus_id] += gen_val
        
    # 2. Build injection arrays sorted by bus ID
    g_array = np.array([bus_gen_totals[b] for b in sorted(load_vector.keys())])
    d_array = np.array([load_vector[b]/baseMVA for b in sorted(load_vector.keys())])
    
    # 3. Calculate flows: f = PTDF * (P_gen - P_demand)
    net_injections = g_array - d_array
    flows = PTDF_matrix.dot(net_injections)
    rateA = branch_df['rateA'].values / baseMVA
    
    max_violation = 0.0
    worst_line_idx = None
    
    # 4. Check against thermal limits
    for l, flow in enumerate(flows):
        limit = rateA[l]
        if limit > 0: 
            violation = abs(flow) - limit
            if violation > max_violation:
                max_violation = violation
                worst_line_idx = branch_df.iloc[l]['line_ID']
                
    return max_violation, worst_line_idx

def build_and_solve_ccga_master(bus_df, gen_df, branch_df, cost_df, load_vector, active_S, baseMVA=100.0):
    """Builds the CCGA Master Problem, adding complex constraints ONLY for active_S."""
    model = pyo.ConcreteModel()
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    model.All_Contingencies = pyo.Set(initialize=model.Gens)
    model.Active_S = pyo.Set(initialize=active_S) 

    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    c1 = dict(zip(cost_df['gen_ID'], cost_df['c1']))
    c2 = dict(zip(cost_df['gen_ID'], cost_df['c2']))
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    
    # Extract gamma dynamically, defaulting to 1.0 if not found
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

    # ==========================================
    # 1. NOMINAL STATE (Base Case)
    # ==========================================
    # Variables without hard bounds; bounds enforced via constraints & slacks
    model.Pg = pyo.Var(model.Gens) 
    model.Theta = pyo.Var(model.Buses, bounds=(-math.pi, math.pi))
    model.Pf = pyo.Var(model.Branches)
    
    # SLACK VARIABLES (Non-Negative)
    model.Sl_nom_line = pyo.Var(model.Branches, within=pyo.NonNegativeReals)
    model.Sl_Pg_nom_up = pyo.Var(model.Gens, within=pyo.NonNegativeReals)
    model.Sl_Pg_nom_down = pyo.Var(model.Gens, within=pyo.NonNegativeReals)

    # 2. PROVISIONAL CONTINGENCY VARIABLES (For all s)
    model.Pgs_prov = pyo.Var(model.All_Contingencies, model.Gens)
    model.Sl_Pg_prov_up = pyo.Var(model.All_Contingencies, model.Gens, within=pyo.NonNegativeReals)
    model.Sl_Pg_prov_down = pyo.Var(model.All_Contingencies, model.Gens, within=pyo.NonNegativeReals)

    # 3. EXACT ACTIVE CONTINGENCIES (Only for s in Active_S)
    model.ns = pyo.Var(model.Active_S, bounds=(0, 1))
    model.xs = pyo.Var(model.Active_S, model.Gens, domain=pyo.Binary)
    model.Thetas = pyo.Var(model.Active_S, model.Buses, bounds=(-math.pi, math.pi))
    model.Pfs = pyo.Var(model.Active_S, model.Branches)
    model.Sl_cont_line = pyo.Var(model.Active_S, model.Branches, within=pyo.NonNegativeReals)

    # --- OBJECTIVE FUNCTION ---
    def obj_rule(m):
        penalty = 1e5 # Heavy penalty to ensure slacks are only used when mathematically necessary
        
        gen_cost = sum(c2[i] * ((m.Pg[i] * baseMVA)**2) + c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
        
        slack_cost_nom = penalty * (
            sum(m.Sl_nom_line[l] for l in m.Branches) + 
            sum(m.Sl_Pg_nom_up[i] + m.Sl_Pg_nom_down[i] for i in m.Gens)
        )
        
        slack_cost_prov = penalty * sum(m.Sl_Pg_prov_up[s, i] + m.Sl_Pg_prov_down[s, i] for s in m.All_Contingencies for i in m.Gens)
        
        if len(active_S) > 0:
            slack_cost_cont = penalty * sum(m.Sl_cont_line[s, l] for s in m.Active_S for l in m.Branches)
        else:
            slack_cost_cont = 0.0
            
        return gen_cost + slack_cost_nom + slack_cost_prov + slack_cost_cont
        
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- NOMINAL CONSTRAINTS ---
    def flow_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf[l] == (m.Theta[bus_from] - m.Theta[bus_to]) / x[l]
    model.flow_eq = pyo.Constraint(model.Branches, rule=flow_rule)

    def limit_upper_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pf[l] - m.Sl_nom_line[l] <= rateA[l]
    model.limit_upper_eq = pyo.Constraint(model.Branches, rule=limit_upper_rule)

    def limit_lower_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pf[l] + m.Sl_nom_line[l] >= -rateA[l]
    model.limit_lower_eq = pyo.Constraint(model.Branches, rule=limit_lower_rule)

    def pg_upper_rule(m, i):
        return m.Pg[i] - m.Sl_Pg_nom_up[i] <= pmax[i]
    model.pg_upper_eq = pyo.Constraint(model.Gens, rule=pg_upper_rule)

    def pg_lower_rule(m, i):
        return m.Pg[i] + m.Sl_Pg_nom_down[i] >= pmin[i]
    model.pg_lower_eq = pyo.Constraint(model.Gens, rule=pg_lower_rule)

    def balance_rule(m, b):
        gen_total = sum(m.Pg[g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pf[l] for l in lines_from[b])
        flow_in = sum(m.Pf[l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)
    model.ref_bus = pyo.Constraint(expr=model.Theta[ref_bus_id] == 0)

    # --- PROVISIONAL CONSTRAINTS ---
    def failed_gen_prov_rule(m, s):
        return m.Pgs_prov[s, s] == 0
    model.failed_gen_prov_eq = pyo.Constraint(model.All_Contingencies, rule=failed_gen_prov_rule)

    def global_demand_prov_rule(m, s):
        total_load = sum(load_vector[b] for b in m.Buses) / baseMVA
        return sum(m.Pgs_prov[s, i] for i in m.Gens) == total_load
    model.global_demand_prov_eq = pyo.Constraint(model.All_Contingencies, rule=global_demand_prov_rule)
    
    def provisional_bound_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] - m.Pg[i] <= gamma[i] * pmax[i]
    model.provisional_bound_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=provisional_bound_rule)

    def pgs_prov_upper_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] - m.Sl_Pg_prov_up[s, i] <= pmax[i]
    model.pgs_prov_upper_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=pgs_prov_upper_rule)

    def pgs_prov_lower_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] + m.Sl_Pg_prov_down[s, i] >= 0
    model.pgs_prov_lower_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=pgs_prov_lower_rule)

    # --- ACTIVE CONTINGENCY CONSTRAINTS ---
    def apr_upper_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * pmax[i] <= pmax[i] * (1 - m.xs[s, i])
    model.apr_upper_eq = pyo.Constraint(model.Active_S, model.Gens, rule=apr_upper_rule)

    def apr_lower_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * pmax[i] >= -pmax[i] * (1 - m.xs[s, i])
    model.apr_lower_eq = pyo.Constraint(model.Active_S, model.Gens, rule=apr_lower_rule)

    def apr_limit_rule1(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pg[i] + m.ns[s] * gamma[i] * pmax[i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_limit_eq1 = pyo.Constraint(model.Active_S, model.Gens, rule=apr_limit_rule1)

    def apr_limit_rule2(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_limit_eq2 = pyo.Constraint(model.Active_S, model.Gens, rule=apr_limit_rule2)

    def flow_cont_rule(m, s, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pfs[s, l] == (m.Thetas[s, bus_from] - m.Thetas[s, bus_to]) / x[l]
    model.flow_cont_eq = pyo.Constraint(model.Active_S, model.Branches, rule=flow_cont_rule)

    def limit_cont_upper_rule(m, s, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pfs[s, l] - m.Sl_cont_line[s, l] <= rateA[l]
    model.limit_cont_upper_eq = pyo.Constraint(model.Active_S, model.Branches, rule=limit_cont_upper_rule)

    def limit_cont_lower_rule(m, s, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return m.Pfs[s, l] + m.Sl_cont_line[s, l] >= -rateA[l]
    model.limit_cont_lower_eq = pyo.Constraint(model.Active_S, model.Branches, rule=limit_cont_lower_rule)

    def balance_cont_rule(m, s, b):
        gen_total = sum(m.Pgs_prov[s, g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pfs[s, l] for l in lines_from[b])
        flow_in = sum(m.Pfs[s, l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_cont_eq = pyo.Constraint(model.Active_S, model.Buses, rule=balance_cont_rule)
    
    def ref_bus_cont_rule(m, s):
        return m.Thetas[s, ref_bus_id] == 0
    model.ref_bus_cont_eq = pyo.Constraint(model.Active_S, rule=ref_bus_cont_rule)

    # --- SOLVE ---
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(model, tee=False)
    
    # Failsafe: Should never hit due to slacks, but protects the loop if Gurobi license drops, etc.
    if results.solver.termination_condition == TerminationCondition.infeasible or \
       results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded:
        raise ValueError(f"Model mathematically infeasible. Check grid data format.")
        
    optimal_g = {i: pyo.value(model.Pg[i]) * baseMVA for i in model.Gens}
    return optimal_g, results.solver.termination_condition

def run_ccga_algorithm(bus_df, gen_df, branch_df, cost_df, load_vector, PTDF_matrix):
    """Executes the CCGA Loop."""
    active_S = [] 
    epsilon = 0.05 / 100.0 # Tolerance converted to per-unit
    iteration = 0
    
    bus_gen_map = gen_df.set_index('gen_ID')['bus_i'].to_dict()
    
    while True:
        optimal_g, status = build_and_solve_ccga_master(
            bus_df, gen_df, branch_df, cost_df, load_vector, active_S
        )
        
        global_max_violation = 0.0
        worst_contingency = None
        worst_line = None
        
        for s in gen_df['gen_ID']:
            n_s, g_s = calculate_primary_response(optimal_g, s, gen_df, load_vector)
            
            max_viol, line_idx = check_contingency_violations(g_s, PTDF_matrix, load_vector, branch_df, bus_gen_map)
            
            if max_viol > global_max_violation:
                global_max_violation = max_viol
                worst_contingency = s
                worst_line = line_idx
                
        if global_max_violation <= epsilon:
            break
            
        if worst_contingency not in active_S:
            active_S.append(worst_contingency)
            print(f"      -> Iteration {iteration}: Line {worst_line} violated under loss of Gen {worst_contingency}. Adding to Master.")
        else:
            print(f"      -> Iteration {iteration}: Line {worst_line} under loss of Gen {worst_contingency} cannot be physically secured. Slack variable active. Ending loop.")
            break 
        
        iteration += 1
        
    return optimal_g, status, iteration, active_S