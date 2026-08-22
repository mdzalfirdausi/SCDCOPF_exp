import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. DATA SPLITTING & PREP
# ==========================================
def create_zonal_data(case, zone1_buses, zone2_buses):
    zonal_data = {'zone1': {}, 'zone2': {}, 'tie_lines': [], 'boundary_buses': [], 'global_Kg': []}
    
    # Define Global Contingencies (All generators in the system)
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
# 2. LOCAL ZONE MODEL BUILDER
# ==========================================
def build_admm_zone(zone_id, zone_data, full_branch_df, tie_lines, boundary_buses, is_reference_zone=False, ref_bus_id=None):
    model = pyo.ConcreteModel(name=f"Zone_{zone_id}")
    
    # Sets
    model.LocalBuses = pyo.Set(initialize=[int(x) for x in zone_data['bus']['bus_i'].tolist()])
    neighbor_buses = [int(b) for b in boundary_buses if b not in model.LocalBuses]
    model.NeighborBuses = pyo.Set(initialize=neighbor_buses)
    model.AllBuses = model.LocalBuses | model.NeighborBuses
    model.BoundaryBuses = pyo.Set(initialize=[int(b) for b in boundary_buses])
    
    model.Gens = pyo.Set(initialize=[int(x) for x in zone_data['gen']['gen_ID'].tolist()])
    model.LocalBranches = pyo.Set(initialize=[int(x) for x in zone_data['branch']['line_ID'].tolist()])
    model.TieLines = pyo.Set(initialize=[int(x) for x in tie_lines])
    model.AllBranches = model.LocalBranches | model.TieLines
    
    # Global Contingency Sets
    model.Kg_Global = pyo.Set(initialize=zone_data['global_Kg'])
    model.Kg_and_Base = pyo.Set(initialize=['base'] + zone_data['global_Kg'])
    
    baseMVA = zone_data['baseMVA']['baseMVA'][0]
    gamma = zone_data['gamma']
    
    pmax = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmax'] / baseMVA))
    pmin = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmin'] / baseMVA))
    gcap = {i: pmax[i] - pmin[i] for i in model.Gens}
    
    gencost = zone_data['gencost'][['c2', 'c1', 'c0']].values
    c2 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 0] * baseMVA**2))
    c1 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 1] * baseMVA))
    c0 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 2]))
    Pd = dict(zip(zone_data['bus']['bus_i'], zone_data['bus']['Pd'] / baseMVA))
    
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

    # ADMM Parameters
    model.rho = pyo.Param(initialize=zone_data['rho_ADMM'], mutable=True)
    # Phase Angle Consensus
    model.u_va = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    model.Va_target = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    # Global Signal Consensus (zk)
    model.u_zk = pyo.Param(model.Kg_Global, initialize=0.0, mutable=True)
    model.neighbor_zk = pyo.Param(model.Kg_Global, initialize=0.0, mutable=True)

    # VARIABLES: Base Case
    model.Pg_base = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Va_base = pyo.Var(model.AllBuses, bounds=(-np.pi, np.pi))
    model.Pf_base = pyo.Var(model.AllBranches)
    model.eta_base = pyo.Var(model.AllBranches, domain=pyo.NonNegativeReals)

    # VARIABLES: Contingencies
    model.Pg_k = pyo.Var(model.Kg_Global, model.Gens, bounds=lambda m, k, i: (pmin[i], pmax[i]))
    model.Va_k = pyo.Var(model.Kg_Global, model.AllBuses, bounds=(-np.pi, np.pi))
    model.Pf_k = pyo.Var(model.Kg_Global, model.AllBranches)
    model.eta_k = pyo.Var(model.Kg_Global, model.AllBranches, domain=pyo.NonNegativeReals)
    
    # VARIABLES: APR (Binary & Global Signal)
    model.xk = pyo.Var(model.Kg_Global, model.Gens, domain=pyo.Binary)
    model.zk = pyo.Var(model.Kg_Global, bounds=(0, 1))

    # Helper function to grab the correct phase angle for objective
    def get_va(m, k, b):
        return m.Va_base[b] if k == 'base' else m.Va_k[k,b]

    # --- Objective ---
    def obj_rule(m):
        gen_cost = sum(c2[i] * m.Pg_base[i]**2 + c1[i] * m.Pg_base[i] + c0[i] for i in m.Gens)
        penalty_cost = zone_data['M_eta'] * (
            sum(m.eta_base[l] for l in m.AllBranches) + 
            sum(m.eta_k[k, l] for k in m.Kg_Global for l in m.AllBranches)
        )
        
        # ADMM Penalty: Force agreement on Phase Angles (all states) AND zk (contingency states)
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
    
    # --- Base Case Constraints ---
    def flow_base_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_base[l] == susceptance[l] * (m.Va_base[bus_from] - m.Va_base[bus_to])
    model.flow_base_eq = pyo.Constraint(model.AllBranches, rule=flow_base_rule)

    def balance_base_rule(m, b):
        gen_total = sum(m.Pg_base[g] for g in bus_gens[b])
        flow_out = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][1] == b)
        return gen_total - Pd[b] == flow_out - flow_in
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

    # --- Contingency Constraints (APR) ---
    def failed_gen_rule(m, k, i):
        if k == i: return m.Pg_k[k, i] == 0.0 # Failed gen drops to 0
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

    # Post-Contingency Network
    def flow_k_rule(m, k, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_k[k, l] == susceptance[l] * (m.Va_k[k, bus_from] - m.Va_k[k, bus_to])
    model.flow_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=flow_k_rule)

    def balance_k_rule(m, k, b):
        gen_total = sum(m.Pg_k[k, g] for g in bus_gens[b])
        flow_out = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][1] == b)
        return gen_total - Pd[b] == flow_out - flow_in
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
# 3. ADMM COORDINATOR (SAFE MULTI-CORE)
# ==========================================
def run_distributed_admm(case, zone1_buses, zone2_buses, max_iters=100, tol=5e-3):
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    tie_lines = zonal_data['tie_lines']
    boundary_buses = zonal_data['boundary_buses']
    global_Kg = zonal_data['global_Kg']
    kg_and_base = ['base'] + global_Kg
    
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    is_ref_z1 = ref_bus in zone1_buses
    
    print(f"Building MILP models...\nTie-Lines: {tie_lines}\nBoundary Buses: {boundary_buses}\nGlobal Contingencies: {global_Kg}")
    model_z1 = build_admm_zone(1, zonal_data['zone1'], case['branch'], tie_lines, boundary_buses, is_ref_z1, ref_bus)
    model_z2 = build_admm_zone(2, zonal_data['zone2'], case['branch'], tie_lines, boundary_buses, not is_ref_z1, ref_bus)
    
    # --- HARDWARE OPTIMIZATION ---
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    print(f"\nHardware Allocation: {total_cores} total cores detected. Gurobi will use ALL cores per solve.")

    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['Threads'] = total_cores  # Unleash all 32 cores on the active zone
    
    # ADMM States
    u_va = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses} 
    Va_z1 = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    Va_z2 = {(k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    
    u_zk = {k: 0.0 for k in global_Kg}
    zk_z1 = {k: 0.0 for k in global_Kg}
    zk_z2 = {k: 0.0 for k in global_Kg}
    
    rho = case['rho_ADMM']
    
    print("\nStarting ADMM Loop...")
    for itr in range(1, max_iters + 1):
        
        # We turn on 'tee=True' just for the very first iteration so you can verify 
        # that Gurobi is actually launching and not hanging on a license check.
        show_log = True if itr == 1 else False
        
        # --- A. Update Zone 1 ---
        for b in boundary_buses:
            for k in kg_and_base:
                model_z1.Va_target[k, b].set_value(Va_z2[k, b])
                model_z1.u_va[k, b].set_value(u_va[k, b])
        for k in global_Kg:
            model_z1.neighbor_zk[k].set_value(zk_z2[k])
            model_z1.u_zk[k].set_value(u_zk[k])
        
        if show_log: print("\n--- Launching Zone 1 Solve (Check Gurobi Log) ---")
        solver.solve(model_z1, tee=show_log)
        
        for b in boundary_buses:
            Va_z1['base', b] = pyo.value(model_z1.Va_base[b])
            for k in global_Kg:
                Va_z1[k, b] = pyo.value(model_z1.Va_k[k, b])
        for k in global_Kg:
            zk_z1[k] = pyo.value(model_z1.zk[k])
            
        # --- B. Update Zone 2 ---
        for b in boundary_buses:
            for k in kg_and_base:
                model_z2.Va_target[k, b].set_value(Va_z1[k, b])
                model_z2.u_va[k, b].set_value(u_va[k, b]) 
        for k in global_Kg:
            model_z2.neighbor_zk[k].set_value(zk_z1[k])
            model_z2.u_zk[k].set_value(u_zk[k])
            
        if show_log: print("\n--- Launching Zone 2 Solve (Check Gurobi Log) ---")
        solver.solve(model_z2, tee=show_log)
        
        for b in boundary_buses:
            Va_z2['base', b] = pyo.value(model_z2.Va_base[b])
            for k in global_Kg:
                Va_z2[k, b] = pyo.value(model_z2.Va_k[k, b])
        for k in global_Kg:
            zk_z2[k] = pyo.value(model_z2.zk[k])

        # --- C. Check Convergence & Update Multipliers ---
        res_va = sum((Va_z1[k, b] - Va_z2[k, b])**2 for k in kg_and_base for b in boundary_buses)
        res_zk = sum((zk_z1[k] - zk_z2[k])**2 for k in global_Kg)
        primal_residual = np.sqrt(res_va + res_zk)
        
        print(f"--- Iteration {itr} --- Primal Residual (Va & zk): {primal_residual:.6f}")
        
        # The "Relax-and-Fix" Heuristic
        if primal_residual <= tol:
            print(f"\nMILP ADMM Converged to integer floor ({primal_residual:.6f} <= {tol}).")
            print("Locking binary variables and running final continuous polishing pass...")
            
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
                Va_z1['base', b] = pyo.value(model_z1.Va_base[b])
                Va_z2['base', b] = pyo.value(model_z2.Va_base[b])
                for k in global_Kg:
                    Va_z1[k, b] = pyo.value(model_z1.Va_k[k, b])
                    Va_z2[k, b] = pyo.value(model_z2.Va_k[k, b])
            for k in global_Kg:
                zk_z1[k] = pyo.value(model_z1.zk[k])
                zk_z2[k] = pyo.value(model_z2.zk[k])
                
            res_va = sum((Va_z1[k, b] - Va_z2[k, b])**2 for k in kg_and_base for b in boundary_buses)
            res_zk = sum((zk_z1[k] - zk_z2[k])**2 for k in global_Kg)
            final_residual = np.sqrt(res_va + res_zk)
            
            print(f"Final Polished Residual: {final_residual:.6f}")
            print("D-SCDCOPF Complete!")
            break
            
        for b in boundary_buses:
            for k in kg_and_base:
                u_va[k, b] = u_va[k, b] + rho * (Va_z1[k, b] - Va_z2[k, b])
        for k in global_Kg:
            u_zk[k] = u_zk[k] + rho * (zk_z1[k] - zk_z2[k])

    return model_z1, model_z2

def save_admm_results(model_z1, model_z2, case_name, base_dir="data"):
    """
    Extracts variables from both zonal models, merges them, 
    and saves to an Excel file with multiple sheets in a specific subdirectory.
    """
    # 1. Define the specific subdirectory path
    output_dir = os.path.join(base_dir, "admm_result")
    
    # 2. Create the directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Construct the final file path
    output_path = os.path.join(output_dir, f"{case_name}_admm_results.xlsx")
    
    print(f"\nExtracting results to {output_path}...")

    # --- 1. Extract Generator Data (Pg_base, Pg_k, xk) ---
    gen_data = []
    for m in [model_z1, model_z2]:
        for i in m.Gens:
            row = {
                'Zone': m.name,
                'Gen_ID': i,
                'Pg_base': pyo.value(m.Pg_base[i], exception=False)
            }
            # Add contingency generation and binary limits
            for k in m.Kg_Global:
                row[f'Pg_k_{k}'] = pyo.value(m.Pg_k[k, i], exception=False)
                row[f'xk_{k}'] = pyo.value(m.xk[k, i], exception=False)
            gen_data.append(row)
            
    df_gen = pd.DataFrame(gen_data).sort_values('Gen_ID')

    # --- 2. Extract Global Signal (zk) ---
    # Since zk reached consensus, we only need to pull it from Zone 1
    zk_data = []
    for k in model_z1.Kg_Global:
        zk_data.append({
            'Contingency_k': k,
            'zk_signal': pyo.value(model_z1.zk[k], exception=False)
        })
    df_zk = pd.DataFrame(zk_data)

    # --- 3. Extract Branch Flows (Pf_base) ---
    # We use a set to avoid duplicating tie-lines that appear in both models
    branch_data = {}
    for m in [model_z1, model_z2]:
        for l in m.AllBranches:
            if l not in branch_data:
                branch_data[l] = {
                    'Branch_ID': l,
                    'Pf_base': pyo.value(m.Pf_base[l], exception=False)
                }
    df_branch = pd.DataFrame(list(branch_data.values())).sort_values('Branch_ID')

    # --- 4. Extract Phase Angles (Va_base) ---
    bus_data = {}
    for m in [model_z1, model_z2]:
        for b in m.AllBuses:
            if b not in bus_data:
                bus_data[b] = {
                    'Bus_ID': b,
                    'Va_base_rad': pyo.value(m.Va_base[b], exception=False)
                }
    df_bus = pd.DataFrame(list(bus_data.values())).sort_values('Bus_ID')

    # --- Save to Excel ---
    with pd.ExcelWriter(output_path) as writer:
        df_gen.to_excel(writer, sheet_name='Generators', index=False)
        df_zk.to_excel(writer, sheet_name='Global_Signal', index=False)
        df_branch.to_excel(writer, sheet_name='Branch_Flows', index=False)
        df_bus.to_excel(writer, sheet_name='Phase_Angles', index=False)
        
    print(f"Success: Results securely saved in the {output_dir} folder!")
# ==========================================
# 4. MAIN EXECUTION BLOCK 
# ==========================================
if __name__ == "__main__":
    case_name = 'pglib_opf_case30_fsr'
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    print(f"Loading data from {case_path}...")
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
    
    zone1 = list(range(1, 16))   
    zone2 = list(range(16, 31))  
    
    model_z1, model_z2 = run_distributed_admm(case, zone1, zone2, max_iters=1000)

    save_admm_results(model_z1, model_z2, case_name)