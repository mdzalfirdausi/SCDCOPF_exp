import pyomo.environ as pyo
import pandas as pd
import math 
import numpy as np

def build_ptdf(bus_df, branch_df, ref_bus_id=69):
    """
    Constructs the Power Transfer Distribution Factor (PTDF) matrix directly from network data.
    """
    # 1. Map bus IDs to sequential matrix indices (0 to N-1) to ensure perfect alignment
    bus_list = sorted(bus_df['bus_i'].tolist())
    num_buses = len(bus_list)
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_list)}
    
    num_branches = len(branch_df)
    
    # 2. Build incidence matrix A (L x N) and diagonal susceptance Bd (L x L)
    A = np.zeros((num_branches, num_buses))
    Bd = np.zeros((num_branches, num_branches))
    
    for idx, row in branch_df.iterrows():
        from_idx = bus_idx[row['bus_i']]
        to_idx = bus_idx[row['bus_j']]
        
        A[idx, from_idx] = 1.0
        A[idx, to_idx] = -1.0
        # Susceptance is the inverse of reactance
        Bd[idx, idx] = 1.0 / row['x']
        
    # 3. Build B_bus = A^T * Bd * A
    B_bus = A.T @ Bd @ A
    
    # 4. Remove reference bus to make the matrix invertible
    ref_idx = bus_idx[ref_bus_id]
    non_ref_indices = [i for i in range(num_buses) if i != ref_idx]
    
    # Extract the reduced B_bus matrix
    B_bus_reduced = B_bus[np.ix_(non_ref_indices, non_ref_indices)]
    
    # Invert the reduced matrix
    X_bus_reduced = np.linalg.inv(B_bus_reduced)
    
    # 5. Reconstruct full X_bus by padding zeros for the reference bus
    X_bus = np.zeros((num_buses, num_buses))
    X_bus[np.ix_(non_ref_indices, non_ref_indices)] = X_bus_reduced
    
    # 6. Calculate PTDF = Bd * A * X_bus
    PTDF = Bd @ A @ X_bus
    
    return PTDF, bus_list

def solve_dc_opf(bus_df, gen_df, branch_df, cost_df, load_vector, baseMVA=100.0):
    # 1. Ensure a fresh model instantiation for every solve iteration
    model = pyo.ConcreteModel()

    # --- Sets ---
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())

    # --- Parameters ---
    # Map generator capacities and costs
    pmax = dict(zip(gen_df['gen_ID'], gen_df['Pmax'] / baseMVA))
    pmin = dict(zip(gen_df['gen_ID'], gen_df['Pmin'] / baseMVA))
    c1 = dict(zip(cost_df['gen_ID'], cost_df['c1']))
    c2 = dict(zip(cost_df['gen_ID'], cost_df['c2']))
    
    # Map branch parameters
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))
    
    # Track network topology mappings for nodal balance
    bus_gens = {b: [] for b in model.Buses}
    for _, row in gen_df.iterrows():
        bus_gens[row['bus_i']].append(row['gen_ID'])
        
    lines_from = {b: [] for b in model.Buses}
    lines_to = {b: [] for b in model.Buses}
    for _, row in branch_df.iterrows():
        lines_from[row['bus_i']].append(row['line_ID'])
        lines_to[row['bus_j']].append(row['line_ID'])

    # --- Variables ---
    model.Pg = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Theta = pyo.Var(model.Buses, bounds=(-math.pi, math.pi)) # <-- Fixed here
    model.Pf = pyo.Var(model.Branches)

    # --- Objective: Minimize Generation Cost ---
    def obj_rule(m):
        return sum(c2[i] * ((m.Pg[i] * baseMVA)**2) + c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
    model.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Constraints ---
    # 1. DC Power Flow (Kirchhoff's Second Law)
    # Map branch endpoints once to avoid Pandas lookups inside rule blocks
    branch_ends = {row['line_ID']: (row['bus_i'], row['bus_j']) for _, row in branch_df.iterrows()}

    # 1. DC Power Flow (Kirchhoff's Second Law)
    def flow_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf[l] == (m.Theta[bus_from] - m.Theta[bus_to]) / x[l]
    model.flow_eq = pyo.Constraint(model.Branches, rule=flow_rule)

    # 2. Thermal Limits
    def limit_rule(m, l):
        # If rateA is 0, it means unconstrained in standard MATPOWER/pglib formats
        if rateA[l] == 0:
            return pyo.Constraint.Skip
        return (-rateA[l], m.Pf[l], rateA[l])
    model.limit_eq = pyo.Constraint(model.Branches, rule=limit_rule)

    # 3. Nodal Power Balance
    def balance_rule(m, b):
        gen_total = sum(m.Pg[g] for g in bus_gens[b])
        # Use the specific load_vector passed for this instance
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pf[l] for l in lines_from[b])
        flow_in = sum(m.Pf[l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)

    # 4. Reference Bus (Bus 69 is the traditional reference in IEEE-118)
    model.ref_bus = pyo.Constraint(expr=model.Theta[69] == 0)

    # ==========================================
    # N-1 CONTINGENCY & APR INTEGRATION
    # ==========================================
    
    # Set of Contingent States (s in S), representing the loss of each generator
    model.Contingencies = pyo.Set(initialize=model.Gens) 

    # --- Contingency Variables ---
    # ns: Global signal mimicking the level of system response for contingency s
    model.ns = pyo.Var(model.Contingencies, bounds=(0, 1))
    
    # Pgs: Post-contingency generation gs
    model.Pgs = pyo.Var(model.Contingencies, model.Gens, bounds=lambda m, s, i: (0, pmax[i]))
    
    # xs: Binary variable indicating if generator i reached its limit under contingency s
    model.xs = pyo.Var(model.Contingencies, model.Gens, domain=pyo.Binary)

    # --- Automatic Primary Response (APR) Constraints ---
    
    # Assume participation factor gamma is proportional to generator capacity
    total_capacity = sum(pmax.values())
    # Assuming generators can deploy up to 100% of remaining capacity for primary response
    gamma = {i: 1.0 for i in model.Gens}
    
    # 1. Failed generator produces zero power
    def failed_gen_rule(m, s):
        return m.Pgs[s, s] == 0
    model.failed_gen_eq = pyo.Constraint(model.Contingencies, rule=failed_gen_rule)

    # 2. Linearized APR deployment for active generators (Absolute value split into upper/lower bounds)
    def apr_upper_rule(m, s, i):
        if s == i:
            return pyo.Constraint.Skip
        return m.Pgs[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * pmax[i] <= pmax[i] * (1 - m.xs[s, i])
    model.apr_upper_eq = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_upper_rule)

    def apr_lower_rule(m, s, i):
        if s == i:
            return pyo.Constraint.Skip
        return m.Pgs[s, i] - m.Pg[i] - m.ns[s] * gamma[i] * pmax[i] >= -pmax[i] * (1 - m.xs[s, i])
    model.apr_lower_eq = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_lower_rule)

    # 3. Capacity limit indicator bounds
    def apr_limit_rule1(m, s, i):
        if s == i:
            return pyo.Constraint.Skip
        return m.Pg[i] + m.ns[s] * gamma[i] * pmax[i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_limit_eq1 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_limit_rule1)

    def apr_limit_rule2(m, s, i):
        if s == i:
            return pyo.Constraint.Skip
        return m.Pgs[s, i] >= pmax[i] * (1 - m.xs[s, i])
    model.apr_limit_eq2 = pyo.Constraint(model.Contingencies, model.Gens, rule=apr_limit_rule2)

    # ==========================================
    # POST-CONTINGENCY NETWORK CONSTRAINTS
    # ==========================================
    
    # Map branch endpoints to ensure the core basis indices are instantiated correctly 
    # for every loop iteration during model construction
    branch_ends = {row['line_ID']: (row['bus_i'], row['bus_j']) for _, row in branch_df.iterrows()}

    # --- Contingency Network Variables ---
    # Thetas: Post-contingency phase angles for each state s
    model.Thetas = pyo.Var(model.Contingencies, model.Buses, bounds=(-math.pi, math.pi)) # <-- Fixed here
    
    # Pfs: Post-contingency branch flows for each state s
    model.Pfs = pyo.Var(model.Contingencies, model.Branches)

    # --- Contingency Network Equations ---
    
    # 1. Post-Contingency DC Power Flow (Kirchhoff's Second Law)
    def flow_contingency_rule(m, s, l):
        # Instantiate the variable indices for every loop evaluation
        bus_from, bus_to = branch_ends[l]
        return m.Pfs[s, l] == (m.Thetas[s, bus_from] - m.Thetas[s, bus_to]) / x[l]
    model.flow_contingency_eq = pyo.Constraint(model.Contingencies, model.Branches, rule=flow_contingency_rule)

    # 2. Post-Contingency Thermal Limits
    def limit_contingency_rule(m, s, l):
        if rateA[l] == 0:
            return pyo.Constraint.Skip
        return (-rateA[l], m.Pfs[s, l], rateA[l])
    model.limit_contingency_eq = pyo.Constraint(model.Contingencies, model.Branches, rule=limit_contingency_rule)

    # 3. Post-Contingency Nodal Power Balance
    def balance_contingency_rule(m, s, b):
        # Compute generation minus demand equals net flow for state s
        gen_total = sum(m.Pgs[s, g] for g in bus_gens[b])
        load_total = load_vector[b] / baseMVA
        flow_out = sum(m.Pfs[s, l] for l in lines_from[b])
        flow_in = sum(m.Pfs[s, l] for l in lines_to[b])
        return gen_total - load_total == flow_out - flow_in
    model.balance_contingency_eq = pyo.Constraint(model.Contingencies, model.Buses, rule=balance_contingency_rule)

    # 4. Post-Contingency Reference Bus (Pinning Bus 69)
    def ref_bus_contingency_rule(m, s):
        return m.Thetas[s, 69] == 0
    model.ref_bus_contingency_eq = pyo.Constraint(model.Contingencies, rule=ref_bus_contingency_rule)

    # --- Solve ---
    solver = pyo.SolverFactory('gurobi_direct')
    results = solver.solve(model, tee=True)

    # Extract optimal dispatch labels
    optimal_g = {i: pyo.value(model.Pg[i]) * baseMVA for i in model.Gens}
    return optimal_g, results.solver.termination_condition