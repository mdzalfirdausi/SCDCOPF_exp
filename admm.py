import numpy as np
import pandas as pd
import pyomo.environ as pyo

def create_zonal_data(case, zone1_buses, zone2_buses):
    """
    Splits the main case dictionary into two separate dictionaries, 
    one for Zone 1 and one for Zone 2, based on the provided bus lists.
    Also identifies tie-lines.
    """
    zonal_data = {'zone1': {}, 'zone2': {}, 'tie_lines': []}
    
    # --- 1. Identify Tie-Lines ---
    branch_df = case['branch']
    for idx, row in branch_df.iterrows():
        f_bus = row['bus_i']
        t_bus = row['bus_j']
        
        if (f_bus in zone1_buses and t_bus in zone2_buses) or \
           (f_bus in zone2_buses and t_bus in zone1_buses):
            zonal_data['tie_lines'].append(row['line_ID'])
            
    # --- 2. Filter Data for Zone 1 ---
    # Buses
    zonal_data['zone1']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone1_buses)].copy()
    # Generators (Assigned to Zone 1 if their bus is in Zone 1)
    zonal_data['zone1']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone1']['gen']['gen_ID'])].copy()
    # Branches (Internal to Zone 1)
    zonal_data['zone1']['branch'] = branch_df[(branch_df['bus_i'].isin(zone1_buses)) & 
                                              (branch_df['bus_j'].isin(zone1_buses))].copy()
    
    # --- 3. Filter Data for Zone 2 ---
    # Buses
    zonal_data['zone2']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone2_buses)].copy()
    # Generators
    zonal_data['zone2']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone2']['gen']['gen_ID'])].copy()
    # Branches (Internal to Zone 2)
    zonal_data['zone2']['branch'] = branch_df[(branch_df['bus_i'].isin(zone2_buses)) & 
                                              (branch_df['bus_j'].isin(zone2_buses))].copy()
    
    # We will pass the full B, K, L matrices and let the local models use only what they need,
    # or recompute local PTDFs. For true distributed ADMM on DC OPF, it's better to use 
    # B-theta formulation locally rather than full-system PTDFs to maintain privacy.
    
    # For this example, let's assume we'll use a nodal formulation (B-theta) locally 
    # to avoid needing the global PTDF matrix inside the local zone.
    
    zonal_data['baseMVA'] = case['baseMVA']
    zonal_data['M_eta'] = case['M_eta']
    zonal_data['rho_ADMM'] = case['rho_ADMM']
    zonal_data['gamma'] = case['gamma']
    
    return zonal_data

def build_admm_zone(zone_id, zone_data, full_branch_df, tie_lines, is_reference_zone=False, ref_bus_id=None):
    """
    Builds the local Pyomo model for a specific zone using B-theta formulation.
    """
    model = pyo.ConcreteModel(name=f"Zone_{zone_id}")
    
    # --- Sets ---
    model.Buses = pyo.Set(initialize=zone_data['bus']['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=zone_data['gen']['gen_ID'].tolist())
    model.LocalBranches = pyo.Set(initialize=zone_data['branch']['line_ID'].tolist())
    model.TieLines = pyo.Set(initialize=tie_lines)
    model.AllBranches = model.LocalBranches | model.TieLines
    
    # We need to define contingencies. For simplicity, let's say Kg are generator contingencies local to this zone.
    model.Kg = pyo.Set(initialize=model.Gens) # Every local gen is a contingency
    model.Kg_and_Base = pyo.Set(initialize=[0] + list(model.Kg)) # 0 represents base case
    
    baseMVA = zone_data['baseMVA']['baseMVA'][0]
    
    # --- Parameters ---
    # Generator Limits & Costs
    pmax = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmax'] / baseMVA))
    pmin = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmin'] / baseMVA))
    
    gencost = zone_data['gencost'][['c2', 'c1', 'c0']].values
    c2 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 0] * baseMVA**2))
    c1 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 1] * baseMVA))
    c0 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 2]))
    
    # Loads
    Pd = dict(zip(zone_data['bus']['bus_i'], zone_data['bus']['Pd'] / baseMVA))
    
    # Branch Parameters (Local and Tie-lines)
    all_branches = full_branch_df[full_branch_df['line_ID'].isin(model.AllBranches)]
    limit = dict(zip(all_branches['line_ID'], all_branches['rateA'] / baseMVA))
    susceptance = dict(zip(all_branches['line_ID'], 1.0 / all_branches['x']))
    
    # Topology mapping
    branch_ends = {row['line_ID']: (row['bus_i'], row['bus_j']) for _, row in all_branches.iterrows()}
    bus_gens = {b: [] for b in model.Buses}
    for _, row in zone_data['gen'].iterrows():
        bus_gens[row['bus_i']].append(row['gen_ID'])

    # --- ADMM Parameters (Updated by Coordinator) ---
    model.rho = pyo.Param(initialize=zone_data['rho_ADMM'], mutable=True)
    # Multiplier u for each tie line and each state k (base + contingencies)
    model.u = pyo.Param(model.Kg_and_Base, model.TieLines, initialize=0.0, mutable=True)
    # Target flow from the neighbor zone
    model.neighbor_P_tie = pyo.Param(model.Kg_and_Base, model.TieLines, initialize=0.0, mutable=True)

    # --- Variables ---
    # Pre-contingency Base Case
    model.Pg_base = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    # Local Phase Angles. We need angles for local buses. 
    # For tie-lines, we only calculate flow based on our end's angle and treat the other end as a parameter, 
    # OR we treat the tie-line flow itself as the consensus variable. Let's use the latter.
    model.Va_base = pyo.Var(model.Buses, bounds=(-np.pi, np.pi))
    model.Pf_base = pyo.Var(model.AllBranches) # Flows on local AND tie-lines
    
    # Slacks for thermal limits (Base case)
    model.eta_base = pyo.Var(model.AllBranches, domain=pyo.NonNegativeReals)

    # --- Objective Function ---
    def obj_rule(m):
        gen_cost = sum(c2[i] * m.Pg_base[i]**2 + c1[i] * m.Pg_base[i] + c0[i] for i in m.Gens)
        penalty_cost = zone_data['M_eta'] * sum(m.eta_base[l] for l in m.AllBranches)
        
        # ADMM Penalty: u * (P_tie_local - P_tie_neighbor) + rho/2 * (P_tie_local - P_tie_neighbor)^2
        # Note: Direction matters. Let's define positive flow as leaving Zone 1 towards Zone 2.
        # Coordinator must handle the signs carefully.
        admm_cost = sum(
            m.u[0, l] * (m.Pf_base[l] - m.neighbor_P_tie[0, l]) + 
            (m.rho / 2) * (m.Pf_base[l] - m.neighbor_P_tie[0, l])**2
            for l in m.TieLines
        )
        # TODO: Add ADMM costs for contingency states k once those variables are defined.
        
        return gen_cost + penalty_cost + admm_cost
        
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Constraints (Base Case) ---
    
    # 1. Local DC Power Flow (Only for local lines)
    def local_flow_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_base[l] == susceptance[l] * (m.Va_base[bus_from] - m.Va_base[bus_to])
    model.local_flow_eq = pyo.Constraint(model.LocalBranches, rule=local_flow_rule)

    # 2. Nodal Balance
    def balance_rule(m, b):
        gen_total = sum(m.Pg_base[g] for g in bus_gens[b])
        # Find lines connected to this bus (both local and tie)
        flow_out = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][1] == b)
        return gen_total - Pd[b] == flow_out - flow_in
    model.balance_eq = pyo.Constraint(model.Buses, rule=balance_rule)

    # 3. Thermal Limits
    def limit_upper_rule(m, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_base[l] <= limit[l] + m.eta_base[l]
    model.limit_upper_eq = pyo.Constraint(model.AllBranches, rule=limit_upper_rule)
    
    def limit_lower_rule(m, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_base[l] >= -limit[l] - m.eta_base[l]
    model.limit_lower_eq = pyo.Constraint(model.AllBranches, rule=limit_lower_rule)

    # 4. Reference Bus (Only applied if this zone contains the system slack bus)
    if is_reference_zone and ref_bus_id in model.Buses:
        model.ref_bus = pyo.Constraint(expr=model.Va_base[ref_bus_id] == 0)

    # =========================================================================
    # ADD CONTINGENCY STATES & BINARY VARIABLES (APR) HERE (Similar to your Q2)
    # =========================================================================
    # You would replicate the variables (Pg_k, Va_k, Pf_k), 
    # binary variables (xk), global signal (zk) for every k in model.Kg.
    # And add the ADMM consensus terms for Pf_k on model.TieLines to the objective.

    return model

def run_distributed_admm(case, zone1_buses, zone2_buses, max_iters=50, tol=1e-3):
    
    # 1. Split Data
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    tie_lines = zonal_data['tie_lines']
    
    # Determine which zone gets the reference bus (e.g., bus 69 in IEEE 118)
    ref_bus = case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0]
    is_ref_z1 = ref_bus in zone1_buses
    
    # 2. Build Models
    model_z1 = build_admm_zone(1, zonal_data['zone1'], case['branch'], tie_lines, is_ref_z1, ref_bus)
    model_z2 = build_admm_zone(2, zonal_data['zone2'], case['branch'], tie_lines, not is_ref_z1, ref_bus)
    
    solver = pyo.SolverFactory('gurobi')
    
    # 3. Initialization
    # Multipliers for tie-lines (Base case only for this example)
    u = {l: 0.0 for l in tie_lines} 
    # Current flow estimates
    P_tie_z1 = {l: 0.0 for l in tie_lines}
    P_tie_z2 = {l: 0.0 for l in tie_lines}
    
    rho = case['rho_ADMM']
    
    print("Starting ADMM Loop...")
    for itr in range(1, max_iters + 1):
        print(f"\n--- Iteration {itr} ---")
        
        # --- A. Update Zone 1 ---
        for l in tie_lines:
            model_z1.neighbor_P_tie[0, l].set_value(P_tie_z2[l])
            model_z1.u[0, l].set_value(u[l])
        
        res1 = solver.solve(model_z1)
        
        # --- B. Update Zone 2 ---
        for l in tie_lines:
            model_z2.neighbor_P_tie[0, l].set_value(P_tie_z1[l])
            # Sign convention: If Z1 thinks flow is positive (leaving Z1), 
            # Z2 should think flow is negative (entering Z2). 
            # The multiplier update must account for this relative direction.
            model_z2.u[0, l].set_value(-u[l]) 
            
        res2 = solver.solve(model_z2)
        
        # --- C. Gather Tie-Line Flows ---
        # Let's assume the branch data defines 'bus_i' as the From end.
        # We need to standardize direction. Let's say positive = Z1 -> Z2.
        for l in tie_lines:
            f_bus = case['branch'].loc[case['branch']['line_ID'] == l, 'bus_i'].values[0]
            
            # Zone 1's perspective of the flow
            val_z1 = pyo.value(model_z1.Pf_base[l])
            if f_bus in zone2_buses: val_z1 = -val_z1 # reverse if line is defined Z2->Z1
            P_tie_z1[l] = val_z1
            
            # Zone 2's perspective of the flow
            val_z2 = pyo.value(model_z2.Pf_base[l])
            if f_bus in zone2_buses: val_z2 = -val_z2
            # Z2's internal Pf variable might be negative if it's receiving, 
            # but we want to compare the absolute magnitude/direction agreed upon.
            # We align P_tie_z2 to the Z1->Z2 standard direction.
            P_tie_z2[l] = -val_z2 if f_bus in zone1_buses else val_z2

        # --- D. Check Convergence & Update Multiplier ---
        primal_residual = np.sqrt(sum((P_tie_z1[l] - P_tie_z2[l])**2 for l in tie_lines))
        print(f"Primal Residual: {primal_residual:.4f}")
        
        if primal_residual <= tol:
            print("ADMM Converged!")
            break
            
        # Update Multiplier (Standard ADMM update)
        for l in tie_lines:
            u[l] = u[l] + rho * (P_tie_z1[l] - P_tie_z2[l])

    # Extract final results
    return model_z1, model_z2