import os
import time
import math
import argparse
import torch
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# Import the GNN architecture and data tools from your training script
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data, create_pyg_dataset

# =============================================================================
# 1. LOCAL MI-ALADIN DECOUPLED NLP SUBPROBLEM (PYOMO)
# =============================================================================
def build_mi_aladin_zone(zone_id, zone_data, full_branch_df, tie_lines, boundary_buses, is_ref_zone=False, ref_bus_id=None):
    """Builds the continuous ALADIN Pyomo model."""
    model = pyo.ConcreteModel(name=f"MI_ALADIN_Zone_{zone_id}")
    
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
    model.Kg_and_Base = pyo.Set(initialize=['base'] + zone_data['global_Kg'])
    
    baseMVA = zone_data['baseMVA']['baseMVA'][0]
    gamma = zone_data.get('gamma', 0.05)
    
    pmax = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmax'] / baseMVA))
    pmin = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmin'] / baseMVA))
    gcap = {i: pmax[i] - pmin[i] for i in model.Gens}
    
    gencost = zone_data['gencost'][['c2', 'c1', 'c0']].values
    c2 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 0] * baseMVA**2))
    c1 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 1] * baseMVA))
    c0 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 2]))
    
    Pd_init = dict(zip(zone_data['bus']['bus_i'], zone_data['bus']['Pd'] / baseMVA))
    model.Pd = pyo.Param(model.LocalBuses, initialize={b: Pd_init[b] for b in model.LocalBuses}, mutable=True)
    
    all_branches = full_branch_df[full_branch_df['line_ID'].isin(model.AllBranches)]
    limit = dict(zip(all_branches['line_ID'], all_branches['rateA'] / baseMVA))
    susceptance = {int(row['line_ID']): (1.0 / row['x'] if row['x'] != 0 else 10000.0) for _, row in all_branches.iterrows()}
    branch_ends = {int(row['line_ID']): (int(row['bus_i']), int(row['bus_j'])) for _, row in all_branches.iterrows()}
    bus_gens = {b: [] for b in model.LocalBuses}
    for _, row in zone_data['gen'].iterrows():
        bus_gens[int(row['bus_i'])].append(int(row['gen_ID']))

    model.rho = pyo.Param(initialize=100.0, mutable=True)
    model.lam_va = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    model.z_va_target = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)

    model.Pg_base = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Va_base = pyo.Var(model.AllBuses, bounds=(-math.pi, math.pi))
    model.Pf_base = pyo.Var(model.AllBranches)
    model.eta_base = pyo.Var(model.AllBranches, domain=pyo.NonNegativeReals)

    model.Pg_k = pyo.Var(model.Kg_Global, model.Gens, bounds=lambda m, k, i: (pmin[i], pmax[i]))
    model.Va_k = pyo.Var(model.Kg_Global, model.AllBuses, bounds=(-math.pi, math.pi))
    model.Pf_k = pyo.Var(model.Kg_Global, model.AllBranches)
    model.eta_k = pyo.Var(model.Kg_Global, model.AllBranches, domain=pyo.NonNegativeReals)
    
    # RELAXED VARIABLES (Will be fixed by GNN)
    model.xk = pyo.Var(model.Kg_Global, model.Gens, domain=pyo.UnitInterval)
    model.zk = pyo.Var(model.Kg_Global, bounds=(0, 1))

    def get_va(m, k, b):
        return m.Va_base[b] if k == 'base' else m.Va_k[k, b]

    sign = 1.0 if zone_id == 1 else -1.0
    def obj_rule(m):
        gen_cost = sum(c2[i] * m.Pg_base[i]**2 + c1[i] * m.Pg_base[i] + c0[i] for i in m.Gens)
        penalty_cost = zone_data.get('M_eta', 1500) * (
            sum(m.eta_base[l] for l in m.AllBranches) + 
            sum(m.eta_k[k, l] for k in m.Kg_Global for l in m.AllBranches)
        )
        dual_va = sum(sign * m.lam_va[k, b] * get_va(m, k, b) for k in m.Kg_and_Base for b in m.BoundaryBuses)
        prox_va = (m.rho / 2.0) * sum((get_va(m, k, b) - m.z_va_target[k, b])**2 for k in m.Kg_and_Base for b in m.BoundaryBuses)
        return gen_cost + penalty_cost + dual_va + prox_va
        
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    def flow_base_rule(m, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_base[l] == susceptance[l] * (m.Va_base[bus_from] - m.Va_base[bus_to])
    model.flow_base_eq = pyo.Constraint(model.AllBranches, rule=flow_base_rule)

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

    if is_ref_zone and ref_bus_id in model.LocalBuses:
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

    def flow_k_rule(m, k, l):
        bus_from, bus_to = branch_ends[l]
        return m.Pf_k[k, l] == susceptance[l] * (m.Va_k[k, bus_from] - m.Va_k[k, bus_to])
    model.flow_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=flow_k_rule)

    def balance_k_rule(m, k, b):
        gen_total = sum(m.Pg_k[k, g] for g in bus_gens[b])
        flow_out = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][0] == b)
        flow_in = sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][1] == b)
        return gen_total - m.Pd[b] == flow_out - flow_in
    model.balance_k_eq = pyo.Constraint(model.Kg_Global, model.LocalBuses, rule=balance_k_rule)

    if is_ref_zone and ref_bus_id in model.LocalBuses:
        model.ref_bus_k = pyo.Constraint(model.Kg_Global, rule=lambda m, k: m.Va_k[k, ref_bus_id] == 0)

    return model

# =============================================================================
# 2. ALADIN MASTER COUPLED EQUALITY QP
# =============================================================================
def damped_bfgs_update(H_k, s_k, y_k):
    s_k = s_k.reshape(-1, 1)
    y_k = y_k.reshape(-1, 1)
    sBs = (s_k.T @ H_k @ s_k).item()
    sy = (s_k.T @ y_k).item()
    
    if sBs < 1e-8: return H_k
    theta = 1.0 if sy >= 0.2 * sBs else (0.8 * sBs) / (sBs - sy)
        
    r_k = theta * y_k + (1.0 - theta) * (H_k @ s_k)
    sr = (s_k.T @ r_k).item()
    
    if sr < 1e-8: return H_k
    term1 = (H_k @ s_k @ s_k.T @ H_k) / sBs
    term2 = (r_k @ r_k.T) / sr
    H_next = H_k - term1 + term2
    
    eigvals, eigvecs = np.linalg.eigh(H_next)
    eigvals = np.maximum(eigvals, 1e-4)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def solve_aladin_master_qp(x1_dict, x2_dict, g1_dict, g2_dict, H1_mat, H2_mat, ordered_keys, boundary_buses, kg_and_base, lambda_dict, mu=1e4):
    master = pyo.ConcreteModel()
    master.Buses = pyo.Set(initialize=boundary_buses)
    master.Kg_and_Base = pyo.Set(initialize=kg_and_base)

    master.dVa1 = pyo.Var(master.Kg_and_Base, master.Buses)
    master.dVa2 = pyo.Var(master.Kg_and_Base, master.Buses)
    master.s_va = pyo.Var(master.Kg_and_Base, master.Buses)
    master.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def get_var1(m, key): return m.dVa1[key[1], key[2]]
    def get_var2(m, key): return m.dVa2[key[1], key[2]]

    def qp_obj_rule(m):
        t1_lin = sum(g1_dict[key] * get_var1(m, key) for key in ordered_keys)
        t1_quad = 0.5 * sum(get_var1(m, ordered_keys[i]) * H1_mat[i, j] * get_var1(m, ordered_keys[j]) for i in range(len(ordered_keys)) for j in range(len(ordered_keys)))
        t2_lin = sum(g2_dict[key] * get_var2(m, key) for key in ordered_keys)
        t2_quad = 0.5 * sum(get_var2(m, ordered_keys[i]) * H2_mat[i, j] * get_var2(m, ordered_keys[j]) for i in range(len(ordered_keys)) for j in range(len(ordered_keys)))
        slack = sum(lambda_dict[('Va', k, b)] * m.s_va[k, b] + (mu / 2.0) * (m.s_va[k, b]**2) for k in m.Kg_and_Base for b in m.Buses)
        return t1_lin + t1_quad + t2_lin + t2_quad + slack
        
    master.obj = pyo.Objective(rule=qp_obj_rule, sense=pyo.minimize)

    def cons_va_rule(m, k, b): return (x1_dict[('Va', k, b)] + m.dVa1[k, b]) - (x2_dict[('Va', k, b)] + m.dVa2[k, b]) == m.s_va[k, b]
    master.cons_va = pyo.Constraint(master.Kg_and_Base, master.Buses, rule=cons_va_rule)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.solve(master)

    d_x1 = {('Va', k, b): pyo.value(master.dVa1[k, b]) for k in kg_and_base for b in boundary_buses}
    d_x2 = {('Va', k, b): pyo.value(master.dVa2[k, b]) for k in kg_and_base for b in boundary_buses}
    lambda_qp = {('Va', k, b): master.dual[master.cons_va[k, b]] for k in kg_and_base for b in boundary_buses}
    s_val = {('Va', k, b): pyo.value(master.s_va[k, b]) for k in kg_and_base for b in boundary_buses}

    return d_x1, d_x2, lambda_qp, s_val

# =============================================================================
# 3. EXECUTION
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML-Accelerated ALADIN")
    parser.add_argument('--case', type=str, required=True, help="e.g., pglib_opf_case14_ieee")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f" PIONEERING ML-ACCELERATED ALADIN ON {device.type.upper()}")
    print(f"=======================================================")

    case_name = args.case
    case_path = f'../excel_outputs/{case_name}.xlsx'
    case = pd.read_excel(case_path, sheet_name=['baseMVA','bus','gen','gencost','branch'])
    baseMVA = case['baseMVA']['baseMVA'][0]
    case['gamma'], case['M_eta'] = 0.05, 1500

    zero_gen_idx = [num for num, i in enumerate(case['gen'].Pmax.values / baseMVA) if (i == 0 and (case['gen'].Pmin.values / baseMVA)[num] == 0) or (case['gen'].Pmin.values / baseMVA)[num] < 0]
    case['gen'].drop(index=zero_gen_idx, inplace=True)
    case['gencost'].drop(index=zero_gen_idx, inplace=True)

    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses = total_buses[:midpoint]
    zone2_buses = total_buses[midpoint:]

    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])
    global_kg = zonal_data['global_Kg']
    kg_and_base = ['base'] + global_kg
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    
    num_boundaries = len(boundary_buses)
    num_global_kg = len(global_kg)
    num_buses_z1 = len(zonal_data['zone1']['bus'])

    # A. LOAD TRAINED GNN MODELS
    print("Loading Trained GNN Agents...")
    gnn_z1 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    # Changed weights_only=False to prevent pickle load errors
    gnn_z1.load_state_dict(torch.load("data/admm_models/zone1_gnn_agent.pth", map_location=device, weights_only=False))
    gnn_z1.eval()

    gnn_z2 = Zone_ADMM_GNN(num_boundaries, num_global_kg).to(device)
    gnn_z2.load_state_dict(torch.load("data/admm_models/zone2_gnn_agent.pth", map_location=device, weights_only=False))
    gnn_z2.eval()

    # B. PULL A LOAD SCENARIO
    csv_path = f'data/{case_name}_generated_loads.csv'
    load_profiles = pd.read_csv(csv_path)
    scenario_idx = 0 
    scenario_row = load_profiles.iloc[scenario_idx].values / baseMVA

    load_z1 = scenario_row[:num_buses_z1].reshape(1, -1)
    load_z2 = scenario_row[num_buses_z1:].reshape(1, -1)

    graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0].to(device)
    graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0].to(device)

    def get_boundary_mask(zone_bus_df, boundary_buses):
        bus_idx_map = {bus_id: i for i, bus_id in enumerate(zone_bus_df['bus_i'].values)}
        mask = torch.zeros(len(bus_idx_map), dtype=torch.bool)
        for b_id in boundary_buses:
            if b_id in bus_idx_map:
                mask[bus_idx_map[b_id]] = True
        return mask

    graph_z1.boundary_mask = get_boundary_mask(zonal_data['zone1']['bus'], boundary_buses).to(device)
    graph_z2.boundary_mask = get_boundary_mask(zonal_data['zone2']['bus'], boundary_buses).to(device)

    graph_z1.batch = torch.zeros(graph_z1.x.size(0), dtype=torch.long, device=device)
    graph_z2.batch = torch.zeros(graph_z2.x.size(0), dtype=torch.long, device=device)

    # C. GNN INFERENCE FOR BINARY MAPPING
    print("Executing GNN Inference for Binary Mapping...")
    gnn_start_time = time.time()
    with torch.no_grad():
        va_t_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
        uva_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
        zk_t_dummy = torch.zeros(1, num_global_kg, device=device)
        uzk_dummy = torch.zeros(1, num_global_kg, device=device)

        _, _, zk_prob_z1 = gnn_z1(graph_z1, va_t_dummy, uva_dummy, zk_t_dummy, uzk_dummy)
        _, _, zk_prob_z2 = gnn_z2(graph_z2, va_t_dummy, uva_dummy, zk_t_dummy, uzk_dummy)

        zk_hard_z1 = (zk_prob_z1 > 0.5).int().squeeze().cpu().numpy()
        zk_hard_z2 = (zk_prob_z2 > 0.5).int().squeeze().cpu().numpy()
        
    gnn_time = time.time() - gnn_start_time
    print(f" -> GNN Mapped {num_global_kg} Binaries in {gnn_time:.4f} seconds.")

    # D. BUILD CONTINUOUS PYOMO MODELS & LOCK BINARIES
    print("Building Continuous ALADIN Models...")
    model_z1 = build_mi_aladin_zone(1, zonal_data['zone1'], case['branch'], zonal_data['tie_lines'], boundary_buses, ref_bus in zone1_buses, ref_bus)
    model_z2 = build_mi_aladin_zone(2, zonal_data['zone2'], case['branch'], zonal_data['tie_lines'], boundary_buses, ref_bus in zone2_buses, ref_bus)

    for b in model_z1.LocalBuses:
        idx = list(zonal_data['zone1']['bus']['bus_i']).index(b)
        model_z1.Pd[b].set_value(load_z1[0, idx])
    for b in model_z2.LocalBuses:
        idx = list(zonal_data['zone2']['bus']['bus_i']).index(b)
        model_z2.Pd[b].set_value(load_z2[0, idx])

    for k_idx, k in enumerate(global_kg):
        # Prevent indexing error by forcing singular numpy arrays into a 1D list shape if needed
        z1_val = zk_hard_z1.item() if zk_hard_z1.size == 1 else zk_hard_z1[k_idx]
        z2_val = zk_hard_z2.item() if zk_hard_z2.size == 1 else zk_hard_z2[k_idx]
        
        for i in model_z1.Gens:
            model_z1.xk[k, i].fix(z1_val)
            model_z1.zk[k].fix(z1_val)
        for i in model_z2.Gens:
            model_z2.xk[k, i].fix(z2_val)
            model_z2.zk[k].fix(z2_val)

    # E. ALADIN CONTINUOUS COORDINATION
    print("\nStarting ALADIN Continuous Optimization...")
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0

    lam = {('Va', k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    z_target = {('Va', k, b): 0.0 for k in kg_and_base for b in boundary_buses}
    
    ordered_keys = [('Va', k, b) for k in kg_and_base for b in boundary_buses]
    dim = len(ordered_keys)
    rho = 100.0
    mu = 1e4
    tol = 1e-4

    aladin_start = time.time()
    for itr in range(1, 30):
        for b in boundary_buses:
            for k in kg_and_base:
                model_z1.lam_va[k, b].set_value(lam[('Va', k, b)])
                model_z1.z_va_target[k, b].set_value(z_target[('Va', k, b)])
                model_z2.lam_va[k, b].set_value(lam[('Va', k, b)])
                model_z2.z_va_target[k, b].set_value(z_target[('Va', k, b)])
                
        solver.solve(model_z1)
        solver.solve(model_z2)
        
        def get_va_val(m, k, b): return pyo.value(m.Va_base[b]) if k == 'base' else pyo.value(m.Va_k[k, b])

        x1 = {('Va', k, b): get_va_val(model_z1, k, b) for k in kg_and_base for b in boundary_buses}
        x2 = {('Va', k, b): get_va_val(model_z2, k, b) for k in kg_and_base for b in boundary_buses}

        g1 = {key: rho * (x1[key] - z_target[key]) + lam[key] for key in ordered_keys}
        g2 = {key: rho * (x2[key] - z_target[key]) - lam[key] for key in ordered_keys}

        x1_vec = np.array([x1[key] for key in ordered_keys])
        x2_vec = np.array([x2[key] for key in ordered_keys])
        g1_vec = np.array([g1[key] for key in ordered_keys])
        g2_vec = np.array([g2[key] for key in ordered_keys])

        if itr == 1:
            H1_mat = np.eye(dim) * rho
            H2_mat = np.eye(dim) * rho
        else:
            H1_mat = damped_bfgs_update(H1_mat, x1_vec - x1_prev, g1_vec - g1_prev)
            H2_mat = damped_bfgs_update(H2_mat, x2_vec - x2_prev, g2_vec - g2_prev)

        x1_prev, g1_prev = x1_vec.copy(), g1_vec.copy()
        x2_prev, g2_prev = x2_vec.copy(), g2_vec.copy()

        d_x1, d_x2, lambda_qp, s_val = solve_aladin_master_qp(x1, x2, g1, g2, H1_mat, H2_mat, ordered_keys, boundary_buses, kg_and_base, lam, mu)

        primal_residual = math.sqrt(sum(v**2 for v in s_val.values()))
        print(f"Iter {itr:02d} | Primal Residual: {primal_residual:.6f}")
        
        if primal_residual <= tol:
            print(f">>> ALADIN Converged in {itr} iterations!")
            break
            
        for key in z_target.keys():
            z_target[key] = 0.5 * ((x1[key] + d_x1[key]) + (x2[key] + d_x2[key]))
            lam[key] = lambda_qp[key]

    print(f"\n>>> Total GNN + ALADIN Solve Time: {time.time() - aladin_start + gnn_time:.4f}s <<<")