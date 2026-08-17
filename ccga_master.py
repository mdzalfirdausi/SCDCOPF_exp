import pyomo.environ as pyo
import pandas as pd
import numpy as np
import math

def calculate_primary_response(optimal_g, contingency_s, gen_df, load_vector, baseMVA=100.0, tolerance=1e-5):
    """
    Algorithm 1: Bisection search to find the global signal n_s and exact g_s.
    """
    total_demand = sum(load_vector.values()) / baseMVA
    
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    gamma = {i: 1.0 for i in gen_df['gen_ID']} # Reserve participation
    
    # Bisection bounds for n_s
    n_low, n_high = 0.0, 1.0
    n_s = 0.5
    g_s = {}

    for _ in range(50): # 50 iterations is more than enough for high precision
        current_generation = 0.0
        
        for i in gen_df['gen_ID']:
            if i == contingency_s:
                g_s[i] = 0.0 # Failed generator
            else:
                # Linear APR model capped at generator limits
                g_s[i] = min(optimal_g[i] + n_s * gamma[i] * pmax[i], pmax[i])
            
            current_generation += g_s[i]
            
        # Check demand balance
        mismatch = current_generation - total_demand
        
        if abs(mismatch) < tolerance:
            break
        elif mismatch < 0:
            n_low = n_s # Need more generation
        else:
            n_high = n_s # Need less generation
            
        n_s = (n_low + n_high) / 2.0
        
    return n_s, g_s

def check_contingency_violations(g_s, PTDF_matrix, load_vector, branch_df, baseMVA=100.0):
    """
    Algebraically calculates post-contingency line flows and finds maximum violations.
    """
    # Convert dictionaries to arrays ordered by bus indices
    g_array = np.array([g_s.get(b, 0) for b in sorted(load_vector.keys())])
    d_array = np.array([load_vector[b]/baseMVA for b in sorted(load_vector.keys())])
    
    # Net injection P = Generation - Demand
    net_injections = g_array - d_array
    
    # Fast algebraic flow calculation: f = PTDF * P
    flows = PTDF_matrix.dot(net_injections)
    
    rateA = branch_df['rateA'].values / baseMVA
    
    max_violation = 0.0
    worst_line_idx = None
    
    # Compare against limits (K1 and K3 from the paper)
    for l, flow in enumerate(flows):
        limit = rateA[l]
        if limit > 0: # If line has a thermal limit
            violation = abs(flow) - limit
            if violation > max_violation:
                max_violation = violation
                worst_line_idx = branch_df.iloc[l]['line_ID']
                
    return max_violation, worst_line_idx

def build_and_solve_ccga_master(bus_df, gen_df, branch_df, cost_df, load_vector, active_S, baseMVA=100.0):
    """
    Builds the CCGA Master Problem.
    active_S: A list of generator IDs that require full binary disjunctions (the 'S' set in the paper).
    """
    model = pyo.ConcreteModel()

    # --- Sets ---
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    
    # All contingencies vs. Active contingencies (S)
    model.All_Contingencies = pyo.Set(initialize=model.Gens)
    model.Active_S = pyo.Set(initialize=active_S) # Only build binaries for these!

    # --- Parameters ---
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    c1 = dict(zip(cost_df['gen_ID'], cost_df['c1']))
    c2 = dict(zip(cost_df['gen_ID'], cost_df['c2']))
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    gamma = {i: 1.0 for i in model.Gens} # Fractional reserve participation

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
    model.Pg = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Theta = pyo.Var(model.Buses, bounds=(-math.pi, math.pi))
    model.Pf = pyo.Var(model.Branches)

    def obj_rule(m):
        return sum(c2[i] * ((m.Pg[i] * baseMVA)**2) + c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def flow_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf[l] == (m.Theta[bus_from] - m.Theta[bus_to]) / x[l]
    model.flow_eq = pyo.Constraint(model.Branches, rule=flow_rule)

    def limit_rule(m, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return (-rateA[l], m.Pf[l], rateA[l]) # Hard constraint, no penalty
    model.limit_eq = pyo.Constraint(model.Branches, rule=limit_rule)

    def balance_rule(m, b):
        gen_total = sum(m.Pg[g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pf[l] for l in lines_from[b])
        flow_in = sum(m.Pf[l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)
    model.ref_bus = pyo.Constraint(expr=model.Theta[69] == 0)

    # ==========================================
    # 2. PROVISIONAL CONTINGENCY VARIABLES (For all s)
    # ==========================================
    # Instead of building the network for all s, CCGA uses provisional g'_s
    model.Pgs_prov = pyo.Var(model.All_Contingencies, model.Gens, bounds=lambda m, s, i: (0, pmax[i]))

    # Failed generator produces zero
    def failed_gen_prov_rule(m, s):
        return m.Pgs_prov[s, s] == 0
    model.failed_gen_prov_eq = pyo.Constraint(model.All_Contingencies, rule=failed_gen_prov_rule)

    # Global demand must be met by provisional generation
    def global_demand_prov_rule(m, s):
        total_load = sum(load_vector[b] for b in m.Buses) / baseMVA
        return sum(m.Pgs_prov[s, i] for i in m.Gens) == total_load
    model.global_demand_prov_eq = pyo.Constraint(model.All_Contingencies, rule=global_demand_prov_rule)
    
    # Valid bound for post-contingency generation: g'_s - g <= r
    def provisional_bound_rule(m, s, i):
        if s == i: return pyo.Constraint.Skip
        return m.Pgs_prov[s, i] - m.Pg[i] <= gamma[i] * pmax[i]
    model.provisional_bound_eq = pyo.Constraint(model.All_Contingencies, model.Gens, rule=provisional_bound_rule)

    # ==========================================
    # 3. EXACT ACTIVE CONTINGENCIES (Only for s in Active_S)
    # ==========================================
    # The heavy binary disjunctions and network constraints are ONLY built if a contingency violates limits
    model.ns = pyo.Var(model.Active_S, bounds=(0, 1))
    model.xs = pyo.Var(model.Active_S, model.Gens, domain=pyo.Binary)
    
    # Post-contingency network for active S
    model.Thetas = pyo.Var(model.Active_S, model.Buses, bounds=(-math.pi, math.pi))
    model.Pfs = pyo.Var(model.Active_S, model.Branches)

    # APR Logic ONLY applied to Active_S
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

    # Network constraints ONLY for Active_S
    def flow_cont_rule(m, s, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pfs[s, l] == (m.Thetas[s, bus_from] - m.Thetas[s, bus_to]) / x[l]
    model.flow_cont_eq = pyo.Constraint(model.Active_S, model.Branches, rule=flow_cont_rule)

    def limit_cont_rule(m, s, l):
        if rateA[l] == 0: return pyo.Constraint.Skip
        return (-rateA[l], m.Pfs[s, l], rateA[l])
    model.limit_cont_eq = pyo.Constraint(model.Active_S, model.Branches, rule=limit_cont_rule)

    def balance_cont_rule(m, s, b):
        gen_total = sum(m.Pgs_prov[s, g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pfs[s, l] for l in lines_from[b])
        flow_in = sum(m.Pfs[s, l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_cont_eq = pyo.Constraint(model.Active_S, model.Buses, rule=balance_cont_rule)
    
    def ref_bus_cont_rule(m, s):
        return m.Thetas[s, 69] == 0
    model.ref_bus_cont_eq = pyo.Constraint(model.Active_S, rule=ref_bus_cont_rule)

    # --- Solve ---
    solver = pyo.SolverFactory('gurobi_direct')
    results = solver.solve(model, tee=False)
    
    optimal_g = {i: pyo.value(model.Pg[i]) * baseMVA for i in model.Gens}
    return optimal_g, model


def run_ccga_algorithm(bus_df, gen_df, branch_df, cost_df, load_vector):
    """
    Executes the CCGA Loop.
    """
    active_S = [] # Initially empty set of complex contingencies
    epsilon = 0.05 # Tolerance for line violation (MW)
    iteration = 0
    
    while True:
        print(f"--- CCGA Iteration {iteration} | Active Contingencies in Master: {len(active_S)} ---")
        
        # 1. Solve Master Problem to get nominal generation g*
        optimal_g, master_model = build_and_solve_ccga_master(
            bus_df, gen_df, branch_df, cost_df, load_vector, active_S
        )
        
        # 2. Subproblem: Numerical Binary Search & PTDF Checking (Conceptual)
        # For every contingency in the network:
        #   a. Use binary search to find n_s based on optimal_g
        #   b. Apply linear equations to find exact g_s
        #   c. Check g_s against transmission line PTDF matrices (K1, K3) to find violations
        
        max_violation = 0.0
        worst_contingency = None
        worst_line = None
        
        # ... [Implement Algorithm 1 Binary Search and Matrix check here] ...
        
        # 3. Convergence Check
        if max_violation <= epsilon:
            print("CCGA Converged! No lines violated.")
            break
            
        # 4. Update Active Set
        if worst_contingency not in active_S:
            active_S.append(worst_contingency)
            print(f"Added {worst_contingency} to Active Set due to line {worst_line} violation.")
        
        iteration += 1
        
    return optimal_g