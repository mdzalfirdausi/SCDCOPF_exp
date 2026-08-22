import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
import multiprocessing
import copy

# ==========================================
# 1. DATA SPLITTING & PREP
# ==========================================
def create_zonal_data(case, zone1_buses, zone2_buses):
    zonal_data = {'zone1': {}, 'zone2': {}, 'tie_lines': [], 'boundary_buses': [], 'global_Kg': []}
    
    zonal_data['global_Kg'] = [int(x) for x in case['gen']['gen_ID'].tolist()]
    
    branch_df = case['branch']
    boundary_buses = set()
    
    for idx, row in branch_df.iterrows():
        f_bus = int(row['bus_i'])
        t_bus = int(row['bus_j'])
        
        if (f_bus in zone1_buses and t_bus in zone2_buses) or \
           (f_bus in zone2_buses and t_bus in zone1_buses):
            zonal_data['tie_lines'].append(int(row['line_ID']))
            boundary_buses.add(f_bus)
            boundary_buses.add(t_bus)
            
    zonal_data['boundary_buses'] = list(boundary_buses)
            
    zonal_data['zone1']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone1_buses)].copy()
    zonal_data['zone1']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone1']['gen']['gen_ID'])].copy()
    zonal_data['zone1']['branch'] = branch_df[(branch_df['bus_i'].isin(zone1_buses)) & 
                                              (branch_df['bus_j'].isin(zone1_buses))].copy()
    
    zonal_data['zone2']['bus'] = case['bus'][case['bus']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gen'] = case['gen'][case['gen']['bus_i'].isin(zone2_buses)].copy()
    zonal_data['zone2']['gencost'] = case['gencost'][case['gencost']['gen_ID'].isin(zonal_data['zone2']['gen']['gen_ID'])].copy()
    zonal_data['zone2']['branch'] = branch_df[(branch_df['bus_i'].isin(zone2_buses)) & 
                                              (branch_df['bus_j'].isin(zone2_buses))].copy()
    
    for z in ['zone1', 'zone2']:
        zonal_data[z]['baseMVA'] = case['baseMVA']
        zonal_data[z]['M_eta'] = case['M_eta']
        zonal_data[z]['rho_ADMM'] = case['rho_ADMM']
        zonal_data[z]['gamma'] = case['gamma']
        zonal_data[z]['global_Kg'] = zonal_data['global_Kg']
    
    return zonal_data

# ==========================================
# 2. LOCAL ZONE MODEL BUILDER (WITH MUTABLE LOADS)
# ==========================================
def build_admm_zone(zone_id, zone_data, full_branch_df, tie_lines, boundary_buses, is_reference_zone=False, ref_bus_id=None):
    model = pyo.ConcreteModel(name=f"Zone_{zone_id}")
    
    model.LocalBuses = pyo.Set(initialize=[int(x) for x in zone_data['bus']['bus_i'].tolist()])
    neighbor_buses = [int(b) for b in boundary_buses if b not in model.LocalBuses]
    model.NeighborBuses = pyo.Set(initialize=neighbor_buses)
    model.AllBuses = model.LocalBuses | model.NeighborBuses
    model.BoundaryBuses = pyo.Set(initialize=[int(b) for b in boundary_buses])
    
    model.Gens = pyo.Set(initialize=[int(x) for x in zone_data['gen']['gen_ID'].tolist()])
    model.LocalBranches = pyo.Set(initialize=[int(x) for x in zone_data['branch']['line_ID'].tolist()])
    model.TieLines = pyo.Set(initialize=[int(x) for x in tie_lines])
    model.AllBranches = model.LocalBranches | model.TieLines
    
    model.Kg_Global = pyo.Set(initialize=zone_data['global_Kg'])
    model.Kg_and_Base = pyo.Set(initialize=[0] + zone_data['global_Kg'])
    
    baseMVA = zone_data['baseMVA']['baseMVA'][0]
    gamma = zone_data['gamma']
    
    pmax = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmax'] / baseMVA))
    pmin = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmin'] / baseMVA))
    gcap = {i: pmax[i] - pmin[i] for i in model.Gens}
    
    gencost = zone_data['gencost'][['c2', 'c1', 'c0']].values
    c2 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 0] * baseMVA**2))
    c1 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 1] * baseMVA))
    c0 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 2]))
    
    # MACHINE LEARNING FIX: Make Load (Pd) a mutable parameter
    Pd_init = dict(zip(zone_data['bus']['bus_i'], zone_data['bus']['Pd'] / baseMVA))
    model.Pd = pyo.Param(model.LocalBuses, initialize={b: Pd_init[b] for b in model.LocalBuses}, mutable=True)
    
    all_branches = full_branch_df[full_branch_df['line_ID'].isin(model.AllBranches)]
    limit = dict(zip(all_branches['line_ID'], all_branches['rateA'] / baseMVA))
    
    susceptance = {}
    for _, row in all_branches.iterrows():
        x_val = row['x']
        susceptance[int(row['line_ID'])] = 1.0 / x_val if x_val != 0 else 10000.0
    
    branch_ends = {int(row['line_ID']): (int(row['bus_i']), int(row['bus_j'])) for _, row in all_branches.iterrows()}
    bus_gens = {b: [] for b in model.LocalBuses}
    for _, row in zone_data['gen'].iterrows():
        bus_gens[int(row['bus_i'])].append(int(row['gen_ID']))

    model.rho = pyo.Param(initialize=zone_data['rho_ADMM'], mutable=True)
    model.u_va = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    model.Va_target = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    model.u_zk = pyo.Param(model.Kg_Global, initialize=0.0, mutable=True)
    model.neighbor_zk = pyo.Param(model.Kg_Global, initialize=0.0, mutable=True)

    model.Pg_base = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Va_base = pyo.Var(model.AllBuses, bounds=(-np.pi, np.pi))
    model.Pf_base = pyo.Var(model.AllBranches)
    model.eta_base = pyo.Var(model.AllBranches, domain=pyo.NonNegativeReals)

    model.Pg_k = pyo.Var(model.Kg_Global, model.Gens, bounds=lambda m, k, i: (pmin[i], pmax[i]))
    model.Va_k = pyo.Var(model.Kg_Global, model.AllBuses, bounds=(-np.pi, np.pi))
    model.Pf_k = pyo.Var(model.Kg_Global, model.AllBranches)
    model.eta_k = pyo.Var(model.Kg_Global, model.AllBranches, domain=pyo.NonNegativeReals)
    
    model.xk = pyo.Var(model.Kg_Global, model.Gens, domain=pyo.Binary)
    model.zk = pyo.Var(model.Kg_Global, bounds=(0, 1))

    def get_va(m, k, b):
        return m.Va_base[b] if k == 0 else m.Va_k[k,b]

    def obj_rule(m):
        gen_cost = sum(c2[i] * m.Pg_base[i]**2 + c1[i] * m.Pg_base[i] + c0[i] for i in m.Gens)
        penalty_cost = zone_data['M_eta'] * (
            sum(m.eta_base[l] for l in m.AllBranches) + 
            sum(m.eta_k[k, l] for k in m.Kg_Global for l in m.AllBranches)
        )
        
        if zone_id == 1:
            admm_va = sum(
                m.u_va[k, b] * (get_va(m, k, b) - m.Va_target[k, b]) + 
                (m.rho / 2) * (get_va(m, k, b) - m.Va_target[k, b])**2
                for k in m.Kg_and_Base for b in m.BoundaryBuses
            )
            admm_zk = sum(
                m.u_zk[k] * (m.zk[k] - m.neighbor_zk[k]) + 
                (m.rho / 2) * (m.zk[k] - m.neighbor_zk[k])**2
                for k in m.Kg_Global
            )
        else:
            admm_va = sum(
                m.u_va[k, b] * (m.Va_target[k, b] - get_va(m, k, b)) + 
                (m.rho / 2) * (m.Va_target[k, b] - get_va(m, k, b))**2
                for k in m.Kg_and_Base for b in m.BoundaryBuses
            )
            admm_zk = sum(
                m.u_zk[k] * (m.neighbor_zk[k] - m.zk[k]) + 
                (m.rho / 2) * (m.neighbor_zk[k] - m.zk[k])**2
                for k in m.Kg_Global
            )
            
        return gen_cost + penalty_cost + admm_va + admm_zk
        
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    def flow_base_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_base[l] == susceptance[l] * (m.Va_base[bus_from] - m.Va_base[bus_to])
    model.flow_base_eq = pyo.Constraint(model.AllBranches, rule=flow_base_rule)

    # Use mutable Pd parameter
    def balance_base_rule(m, b):
        gen_total = sum(m.Pg_base[g] for g in bus_gens[b])
        flow_out = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][1] == b)
        return gen_total - m.Pd[b] == flow_out - flow_in
    model.balance_base_eq = pyo.Constraint(model.LocalBuses, rule=balance_base_rule)

    def limit_upper_base_rule(m, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_base[l] <= limit[l] + m.eta_base[l]
    model.limit_upper_base_eq = pyo.Constraint(model.AllBranches, rule=limit_upper_base_rule)
    
    def limit_lower_base_rule(m, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_base[l] >= -limit[l] - m.eta_base[l]
    model.limit_lower_base_eq = pyo.Constraint(model.AllBranches, rule=limit_lower_base_rule)

    if is_reference_zone and ref_bus_id in model.LocalBuses:
        model.ref_bus_base = pyo.Constraint(expr=model.Va_base[ref_bus_id] == 0)

    def failed_gen_rule(m, k, i):
        if k == i: return m.Pg_k[k, i] == 0.0 
        return pyo.Constraint.Skip
    model.failed_gen_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=failed_gen_rule)

    def apr_c1_rule(m, k, i):
        if k == i: return pyo.Constraint.Skip
        return m.Pg_k[k, i] - m.Pg_base[i] - m.zk[k] * gamma * gcap[i] <= pmax[i] * (1 - m.xk[k, i])
    model.apr_c1_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c1_rule)

    def apr_c2_rule(m, k, i):
        if k == i: return pyo.Constraint.Skip
        return m.Pg_k[k, i] - m.Pg_base[i] - m.zk[k] * gamma * gcap[i] >= -pmax[i] * (1 - m.xk[k, i])
    model.apr_c2_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c2_rule)

    def apr_c3_rule(m, k, i):
        if k == i: return pyo.Constraint.Skip
        return m.Pg_base[i] + m.zk[k] * gamma * gcap[i] >= pmax[i] * (1 - m.xk[k, i])
    model.apr_c3_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c3_rule)

    def apr_c4_rule(m, k, i):
        if k == i: return pyo.Constraint.Skip
        return m.Pg_k[k, i] >= pmax[i] * (1 - m.xk[k, i])
    model.apr_c4_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c4_rule)

    def flow_k_rule(m, k, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_k[k, l] == susceptance[l] * (m.Va_k[k, bus_from] - m.Va_k[k, bus_to])
    model.flow_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=flow_k_rule)

    def balance_k_rule(m, k, b):
        gen_total = sum(m.Pg_k[k, g] for g in bus_gens[b])
        flow_out = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][1] == b)
        # Use mutable Pd parameter
        return gen_total - m.Pd[b] == flow_out - flow_in
    model.balance_k_eq = pyo.Constraint(model.Kg_Global, model.LocalBuses, rule=balance_k_rule)

    def limit_upper_k_rule(m, k, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_k[k, l] <= limit[l] + m.eta_k[k, l]
    model.limit_upper_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=limit_upper_k_rule)
    
    def limit_lower_k_rule(m, k, l):
        if limit[l] == 0: return pyo.Constraint.Skip
        return m.Pf_k[k, l] >= -limit[l] - m.eta_k[k, l]
    model.limit_lower_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=limit_lower_k_rule)

    if is_reference_zone and ref_bus_id in model.LocalBuses:
        model.ref_bus_k = pyo.Constraint(model.Kg_Global, rule=lambda m, k: m.Va_k[k, ref_bus_id] == 0)

    return model

# ==========================================
# 3. FEATURE EXTRACTOR
# ==========================================
def extract_state_vector(u_va, u_zk, Va_z1, Va_z2, zk_z1, zk_z2, boundary_buses, kg_and_base, global_Kg):
    """Flattens the ADMM variables into a single deterministic 1D array for Neural Networks."""
    b_buses = sorted(list(boundary_buses))
    k_base = sorted(list(kg_and_base))
    k_glob = sorted(list(global_Kg))
    
    features = []
    # Append Dual Variables
    features.extend([u_va[k, b] for k in k_base for b in b_buses])
    features.extend([u_zk[k] for k in k_glob])
    
    # Append Primal Variables (Phase Angles)
    features.extend([Va_z1[k, b] for k in k_base for b in b_buses])
    features.extend([Va_z2[k, b] for k in k_base for b in b_buses])
    
    # Append Primal Variables (Global Signal)
    features.extend([zk_z1[k] for k in k_glob])
    features.extend([zk_z2[k] for k in k_glob])
    
    return np.array(features, dtype=np.float32)

# ==========================================
# 4. ADMM DATA GENERATOR
# ==========================================
def solve_scenario_and_extract(model_z1, model_z2, solver, rho, boundary_buses, kg_and_base, global_Kg, seq_length=10, max_iters=200, tol=5e-3):
    """Runs ADMM for a single modified scenario and returns X (sequence) and y (final)."""
    # Reset states for the new scenario
    u_va = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses} 
    Va_z1 = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    Va_z2 = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    u_zk = {k: 0.0 for k in global_Kg}
    zk_z1 = {k: 0.0 for k in global_Kg}
    zk_z2 = {k: 0.0 for k in global_Kg}
    
    # Unfix binary variables if they were locked in the previous scenario
    for k in global_Kg:
        for i in model_z1.Gens:
            model_z1.xk[k, i].unfix()
        for i in model_z2.Gens:
            model_z2.xk[k, i].unfix()
            
    X_sequence = []
    y_final = None

    for itr in range(1, max_iters + 1):
        # Update Zone 1
        for b in boundary_buses:
            for k in kg_and_base:
                model_z1.Va_target[k, b].set_value(Va_z2[k, b])
                model_z1.u_va[k, b].set_value(u_va[k, b])
        for k in global_Kg:
            model_z1.neighbor_zk[k].set_value(zk_z2[k])
            model_z1.u_zk[k].set_value(u_zk[k])
        
        solver.solve(model_z1, tee=False)
        
        for b in boundary_buses:
            Va_z1[0, b] = pyo.value(model_z1.Va_base[b])
            for k in global_Kg:
                Va_z1[k, b] = pyo.value(model_z1.Va_k[k, b])
        for k in global_Kg:
            zk_z1[k] = pyo.value(model_z1.zk[k])
            
        # Update Zone 2
        for b in boundary_buses:
            for k in kg_and_base:
                model_z2.Va_target[k, b].set_value(Va_z1[k, b])
                model_z2.u_va[k, b].set_value(u_va[k, b]) 
        for k in global_Kg:
            model_z2.neighbor_zk[k].set_value(zk_z1[k])
            model_z2.u_zk[k].set_value(u_zk[k])
            
        solver.solve(model_z2, tee=False)
        
        for b in boundary_buses:
            Va_z2[0, b] = pyo.value(model_z2.Va_base[b])
            for k in global_Kg:
                Va_z2[k, b] = pyo.value(model_z2.Va_k[k, b])
        for k in global_Kg:
            zk_z2[k] = pyo.value(model_z2.zk[k])

        # Track Sequence Data
        if itr <= seq_length:
            X_sequence.append(extract_state_vector(u_va, u_zk, Va_z1, Va_z2, zk_z1, zk_z2, boundary_buses, kg_and_base, global_Kg))

        res_va = sum((Va_z1[k, b] - Va_z2[k, b])**2 for k in kg_and_base for b in boundary_buses)
        res_zk = sum((zk_z1[k] - zk_z2[k])**2 for k in global_Kg)
        primal_residual = np.sqrt(res_va + res_zk)
        
        if itr % 10 == 0 or itr == 1:
            print(f"      [ADMM Iter {itr}] Primal Residual: {primal_residual:.6f}")
            
        if primal_residual <= tol:
            # Lock & Polish
            for k in global_Kg:
                for i in model_z1.Gens:
                    val_z1 = pyo.value(model_z1.xk[k, i], exception=False)
                    model_z1.xk[k, i].fix(round(val_z1) if val_z1 is not None else 0)
                for i in model_z2.Gens:
                    val_z2 = pyo.value(model_z2.xk[k, i], exception=False)
                    model_z2.xk[k, i].fix(round(val_z2) if val_z2 is not None else 0)
                    
            solver.solve(model_z1, tee=False)
            solver.solve(model_z2, tee=False)
            
            for b in boundary_buses:
                Va_z1[0, b] = pyo.value(model_z1.Va_base[b])
                Va_z2[0, b] = pyo.value(model_z2.Va_base[b])
                for k in global_Kg:
                    Va_z1[k, b] = pyo.value(model_z1.Va_k[k, b])
                    Va_z2[k, b] = pyo.value(model_z2.Va_k[k, b])
            for k in global_Kg:
                zk_z1[k] = pyo.value(model_z1.zk[k])
                zk_z2[k] = pyo.value(model_z2.zk[k])
                
            y_final = extract_state_vector(u_va, u_zk, Va_z1, Va_z2, zk_z1, zk_z2, boundary_buses, kg_and_base, global_Kg)
            break
            
        for b in boundary_buses:
            for k in kg_and_base:
                u_va[k, b] = u_va[k, b] + rho * (Va_z1[k, b] - Va_z2[k, b])
        for k in global_Kg:
            u_zk[k] = u_zk[k] + rho * (zk_z1[k] - zk_z2[k])

    # Ensure sequence is padded if it converged before seq_length
    while len(X_sequence) < seq_length:
        X_sequence.append(X_sequence[-1])

    return np.array(X_sequence), y_final

# ==========================================
# 5. DATASET GENERATION LOOP
# ==========================================
if __name__ == "__main__":
    # UPDATE: Now pointing to the 118-bus base case
    case_name = 'pglib_opf_case118_ieee'
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    print(f"Loading Base Data from {case_path}...")
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    
    zero_gen_idx = []
    for num, i in enumerate(case['gen'].Pmax.values / baseMVA):
        if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0:
            zero_gen_idx.append(num)
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)
    
    case['gamma'] = 0.05
    case['M_eta'] = 1500
    case['rho_ADMM'] = 10000.0 
    
    # UPDATE: Splitting 118 buses roughly in half
    zone1 = list(range(1, 60))    # Buses 1 through 59
    zone2 = list(range(60, 119))  # Buses 60 through 118

    zonal_data = create_zonal_data(case, zone1, zone2)
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    
    print("Building Base Pyomo Models (Memory Initialization)...")
    model_z1 = build_admm_zone(1, zonal_data['zone1'], case['branch'], zonal_data['tie_lines'], zonal_data['boundary_buses'], ref_bus in zone1, ref_bus)
    model_z2 = build_admm_zone(2, zonal_data['zone2'], case['branch'], zonal_data['tie_lines'], zonal_data['boundary_buses'], ref_bus in zone2, ref_bus)
    
    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['Threads'] = multiprocessing.cpu_count()
    
    # --- LOAD THE SCENARIO CSV ---
    csv_path = 'data/pglib_opf_case118_ieee_generated_loads.csv'
    print(f"Loading Load Profiles from {csv_path}...")
    load_profiles = pd.read_csv(csv_path)
    
    # Limit to 1000 scenarios as requested
    num_scenarios = min(1000, len(load_profiles))
    seq_length = 10
    X_dataset = []
    y_dataset = []
    
    print(f"\nCommencing Simulation of {num_scenarios} Scenarios...")
    
    for s in range(num_scenarios):
        scenario_start_time = time.time()
        print(f"\n=====================================================")
        print(f" STARTING SCENARIO {s + 1}/{num_scenarios}")
        print(f"=====================================================")
        
        # Extract row 's' from the CSV
        scenario_row = load_profiles.iloc[s]
        
        # Inject new loads directly into Zone 1
        for b in model_z1.LocalBuses:
            if str(b) in scenario_row.index:
                new_pd_mw = scenario_row[str(b)]
            elif b in scenario_row.index:
                new_pd_mw = scenario_row[b]
            else:
                new_pd_mw = scenario_row.iloc[int(b)-1]
                
            model_z1.Pd[b].set_value(new_pd_mw / baseMVA)
            
        # Inject new loads directly into Zone 2
        for b in model_z2.LocalBuses:
            if str(b) in scenario_row.index:
                new_pd_mw = scenario_row[str(b)]
            elif b in scenario_row.index:
                new_pd_mw = scenario_row[b]
            else:
                new_pd_mw = scenario_row.iloc[int(b)-1]
                
            model_z2.Pd[b].set_value(new_pd_mw / baseMVA)

        # Solve ADMM for this specific scenario
        X_seq, y_fin = solve_scenario_and_extract(
            model_z1, model_z2, solver, case['rho_ADMM'], 
            zonal_data['boundary_buses'], [0] + zonal_data['global_Kg'], 
            zonal_data['global_Kg'], seq_length=seq_length
        )
        
        X_dataset.append(X_seq)
        y_dataset.append(y_fin)
        
        duration = time.time() - scenario_start_time
        print(f" ---> Scenario {s + 1} Completed in {duration:.2f} seconds.")

    # Save to disk
    os.makedirs('data/ml_dataset', exist_ok=True)
    np.save(f'data/ml_dataset/{case_name}_X_seq.npy', np.array(X_dataset))
    np.save(f'data/ml_dataset/{case_name}_y_final.npy', np.array(y_dataset))
    print("\nDataset generation complete! Tensors saved to data/ml_dataset/")