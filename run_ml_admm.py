import os
import time
import math
import argparse
import torch
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# Import GNN architecture and data tools
from gnn_erdos import Zone_ADMM_GNN, create_zonal_data, create_pyg_dataset

# =============================================================================
# 1. LOCAL CONTINUOUS ADMM SUBPROBLEM
# =============================================================================
def build_mi_admm_zone(zone_id, zone_data, full_branch_df, tie_lines, boundary_buses, active_contingencies, is_ref_zone=False, ref_bus_id=None):
    """Builds the continuous ADMM Pyomo model using ONLY active contingencies."""
    model = pyo.ConcreteModel(name=f"ADMM_Zone_{zone_id}")
    
    model.LocalBuses = pyo.Set(initialize=[int(x) for x in zone_data['bus']['bus_i'].tolist()])
    neighbor_buses = [int(b) for b in boundary_buses if b not in model.LocalBuses]
    model.NeighborBuses = pyo.Set(initialize=neighbor_buses)
    model.AllBuses = model.LocalBuses | model.NeighborBuses
    model.BoundaryBuses = pyo.Set(initialize=[int(b) for b in boundary_buses])
    
    model.Gens = pyo.Set(initialize=[int(x) for x in zone_data['gen']['gen_ID'].tolist()])
    model.LocalBranches = pyo.Set(initialize=[int(x) for x in zone_data['branch']['line_ID'].tolist()])
    model.TieLines = pyo.Set(initialize=[int(x) for x in tie_lines])
    model.AllBranches = model.LocalBranches | model.TieLines
    
    model.Kg_Global = pyo.Set(initialize=active_contingencies)
    model.Kg_and_Base = pyo.Set(initialize=['base'] + active_contingencies)
    
    baseMVA = zone_data['baseMVA']['baseMVA'][0]
    gamma = zone_data.get('gamma', 0.05)
    
    pmax = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmax'] / baseMVA))
    pmin = dict(zip(zone_data['gen']['gen_ID'], zone_data['gen']['Pmin'] / baseMVA))
    gcap = {i: pmax[i] - pmin[i] for i in model.Gens}
    
    gencost = zone_data['gencost'][['c2', 'c1', 'c0']].values
    c2 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 0] * baseMVA**2))
    c1 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 1] * baseMVA))
    c0 = dict(zip(zone_data['gen']['gen_ID'], gencost[:, 2]))
    
    model.Pd = pyo.Param(model.LocalBuses, initialize={b: 0.0 for b in model.LocalBuses}, mutable=True)
    
    all_branches = full_branch_df[full_branch_df['line_ID'].isin(model.AllBranches)]
    limit = dict(zip(all_branches['line_ID'], all_branches['rateA'] / baseMVA))
    susceptance = {int(row['line_ID']): (1.0 / row['x'] if row['x'] != 0 else 10000.0) for _, row in all_branches.iterrows()}
    branch_ends = {int(row['line_ID']): (int(row['bus_i']), int(row['bus_j'])) for _, row in all_branches.iterrows()}
    bus_gens = {b: [] for b in model.LocalBuses}
    for _, row in zone_data['gen'].iterrows(): bus_gens[int(row['bus_i'])].append(int(row['gen_ID']))

    # PURE ADMM PENALTY PARAMETERS
    model.rho = pyo.Param(initialize=10000.0, mutable=True)
    model.u_va = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)
    model.Va_target = pyo.Param(model.Kg_and_Base, model.BoundaryBuses, initialize=0.0, mutable=True)

    model.Pg_base = pyo.Var(model.Gens, bounds=lambda m, i: (pmin[i], pmax[i]))
    model.Va_base = pyo.Var(model.AllBuses, bounds=(-math.pi, math.pi))
    model.Pf_base = pyo.Var(model.AllBranches)
    model.eta_base = pyo.Var(model.AllBranches, domain=pyo.NonNegativeReals)

    model.Pg_k = pyo.Var(model.Kg_Global, model.Gens, bounds=lambda m, k, i: (pmin[i], pmax[i]))
    model.Va_k = pyo.Var(model.Kg_Global, model.AllBuses, bounds=(-math.pi, math.pi))
    model.Pf_k = pyo.Var(model.Kg_Global, model.AllBranches)
    model.eta_k = pyo.Var(model.Kg_Global, model.AllBranches, domain=pyo.NonNegativeReals)
    
    model.xk = pyo.Var(model.Kg_Global, model.Gens, domain=pyo.Binary)

    def get_va(m, k, b): return m.Va_base[b] if k == 'base' else m.Va_k[k, b]
    
    # EXACT ADMM OBJECTIVE FUNCTION
    def obj_rule(m):
        gen_cost = sum(c2[i] * m.Pg_base[i]**2 + c1[i] * m.Pg_base[i] + c0[i] for i in m.Gens)
        penalty_cost = zone_data.get('M_eta', 1500) * (sum(m.eta_base[l] for l in m.AllBranches) + sum(m.eta_k[k, l] for k in m.Kg_Global for l in m.AllBranches))
        
        if zone_id == 1:
            admm_va = sum(m.u_va[k, b] * (get_va(m, k, b) - m.Va_target[k, b]) + (m.rho / 2.0) * (get_va(m, k, b) - m.Va_target[k, b])**2 for k in m.Kg_and_Base for b in m.BoundaryBuses)
        else:
            admm_va = sum(m.u_va[k, b] * (m.Va_target[k, b] - get_va(m, k, b)) + (m.rho / 2.0) * (m.Va_target[k, b] - get_va(m, k, b))**2 for k in m.Kg_and_Base for b in m.BoundaryBuses)
            
        return gen_cost + penalty_cost + admm_va
        
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    def flow_base_rule(m, l): return m.Pf_base[l] == susceptance[l] * (m.Va_base[branch_ends[l][0]] - m.Va_base[branch_ends[l][1]])
    model.flow_base_eq = pyo.Constraint(model.AllBranches, rule=flow_base_rule)

    def balance_base_rule(m, b): return sum(m.Pg_base[g] for g in bus_gens[b]) - m.Pd[b] == sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][0] == b) - sum(m.Pf_base[l] for l in m.AllBranches if branch_ends[l][1] == b)
    model.balance_base_eq = pyo.Constraint(model.LocalBuses, rule=balance_base_rule)

    def limit_upper_base_rule(m, l): return m.Pf_base[l] <= limit[l] + m.eta_base[l] if limit[l] > 0 else pyo.Constraint.Skip
    def limit_lower_base_rule(m, l): return m.Pf_base[l] >= -limit[l] - m.eta_base[l] if limit[l] > 0 else pyo.Constraint.Skip
    model.limit_upper_base_eq = pyo.Constraint(model.AllBranches, rule=limit_upper_base_rule)
    model.limit_lower_base_eq = pyo.Constraint(model.AllBranches, rule=limit_lower_base_rule)

    if is_ref_zone and ref_bus_id in model.LocalBuses:
        model.ref_bus_base = pyo.Constraint(expr=model.Va_base[ref_bus_id] == 0)

    def failed_gen_rule(m, k, i): return m.Pg_k[k, i] == 0.0 if k == i else pyo.Constraint.Skip
    model.failed_gen_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=failed_gen_rule)

    def apr_c1_rule(m, k, i): return m.Pg_k[k, i] - m.Pg_base[i] - gamma * gcap[i] <= pmax[i] * (1 - m.xk[k, i]) if k != i else pyo.Constraint.Skip
    def apr_c2_rule(m, k, i): return m.Pg_k[k, i] - m.Pg_base[i] - gamma * gcap[i] >= -pmax[i] * (1 - m.xk[k, i]) if k != i else pyo.Constraint.Skip
    model.apr_c1_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c1_rule)
    model.apr_c2_eq = pyo.Constraint(model.Kg_Global, model.Gens, rule=apr_c2_rule)

    def flow_k_rule(m, k, l): return m.Pf_k[k, l] == susceptance[l] * (m.Va_k[k, branch_ends[l][0]] - m.Va_k[k, branch_ends[l][1]])
    model.flow_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=flow_k_rule)

    def balance_k_rule(m, k, b): return sum(m.Pg_k[k, g] for g in bus_gens[b]) - m.Pd[b] == sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][0] == b) - sum(m.Pf_k[k, l] for l in m.AllBranches if branch_ends[l][1] == b)
    model.balance_k_eq = pyo.Constraint(model.Kg_Global, model.LocalBuses, rule=balance_k_rule)

    def limit_upper_k_rule(m, k, l): return m.Pf_k[k, l] <= limit[l] + m.eta_k[k, l] if limit[l] > 0 else pyo.Constraint.Skip
    def limit_lower_k_rule(m, k, l): return m.Pf_k[k, l] >= -limit[l] - m.eta_k[k, l] if limit[l] > 0 else pyo.Constraint.Skip
    model.limit_upper_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=limit_upper_k_rule)
    model.limit_lower_k_eq = pyo.Constraint(model.Kg_Global, model.AllBranches, rule=limit_lower_k_rule)

    if is_ref_zone and ref_bus_id in model.LocalBuses:
        model.ref_bus_k = pyo.Constraint(model.Kg_Global, rule=lambda m, k: m.Va_k[k, ref_bus_id] == 0)

    return model

# =============================================================================
# 2. STANDARDIZED MAIN FUNCTION TO IMPORT
# =============================================================================
def run_ml_admm_scenario(case, zonal_data, load_vector, gnn_z1, gnn_z2, device, baseMVA, verbose=False):
    """Callable function that runs the complete ML + ADMM pipeline."""
    start_ml = time.time()
    
    bus_list = sorted(case['bus']['bus_i'].tolist())
    zone1_buses = zonal_data['zone1']['bus']['bus_i'].tolist()
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    global_kg = zonal_data['global_Kg']
    num_boundaries, num_global_kg = len(boundary_buses), len(global_kg)
    ref_bus = int(case['bus'].loc[case['bus']['type'] == 3, 'bus_i'].values[0])

    scenario_row_pu = np.array([load_vector[b] for b in bus_list]) / baseMVA
    load_z1 = scenario_row_pu[:len(zone1_buses)].reshape(1, -1)
    load_z2 = scenario_row_pu[len(zone1_buses):].reshape(1, -1)

    graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0].to(device)
    graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0].to(device)
    
    # 1. GNN Inference (Acting as the Integer Oracle)
    with torch.no_grad():
        va_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
        zk_dummy = torch.zeros(1, num_global_kg, device=device)
        _, _, zk_prob_z1 = gnn_z1(graph_z1, va_dummy, va_dummy, zk_dummy, zk_dummy)
        _, _, zk_prob_z2 = gnn_z2(graph_z2, va_dummy, va_dummy, zk_dummy, zk_dummy)
        zk_hard_z1 = (zk_prob_z1 > 0.5).int().squeeze().cpu().numpy()
        zk_hard_z2 = (zk_prob_z2 > 0.5).int().squeeze().cpu().numpy()

    active_k_list = []
    for k_idx, k in enumerate(global_kg):
        z1_val = zk_hard_z1.item() if zk_hard_z1.size == 1 else zk_hard_z1[k_idx]
        z2_val = zk_hard_z2.item() if zk_hard_z2.size == 1 else zk_hard_z2[k_idx]
        if z1_val == 1 or z2_val == 1: active_k_list.append(k)

    current_kg_and_base = ['base'] + active_k_list

    # 2. Build Continuous ADMM Models
    model_z1 = build_mi_admm_zone(1, zonal_data['zone1'], case['branch'], zonal_data['tie_lines'], boundary_buses, active_k_list, ref_bus in zone1_buses, ref_bus)
    model_z2 = build_mi_admm_zone(2, zonal_data['zone2'], case['branch'], zonal_data['tie_lines'], boundary_buses, active_k_list, ref_bus not in zone1_buses, ref_bus)

    for b in model_z1.LocalBuses: model_z1.Pd[b].set_value(load_vector[b] / baseMVA)
    for b in model_z2.LocalBuses: model_z2.Pd[b].set_value(load_vector[b] / baseMVA)

    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0

    # 3. GLOBAL CONSENSUS ADMM WITH RESIDUAL BALANCING
    lam_z1 = {(k, b): 0.0 for k in current_kg_and_base for b in boundary_buses}
    lam_z2 = {(k, b): 0.0 for k in current_kg_and_base for b in boundary_buses}
    z_global = {(k, b): 0.0 for k in current_kg_and_base for b in boundary_buses}
    
    rho = 10000.0  
    model_z1.rho.set_value(rho)
    model_z2.rho.set_value(rho)
    
    tol = 1e-4
    ml_iters = 0
    scenario_residuals = []

    for itr in range(1, 100):
        ml_iters = itr
        
        # --- Solve Zone 1 ---
        for b in boundary_buses:
            for k in current_kg_and_base:
                model_z1.Va_target[k, b].set_value(z_global[k, b])
                model_z1.u_va[k, b].set_value(lam_z1[k, b])
        solver.solve(model_z1)
        
        Va_z1 = {}
        for b in boundary_buses:
            Va_z1['base', b] = pyo.value(model_z1.Va_base[b])
            for k in active_k_list:
                Va_z1[k, b] = pyo.value(model_z1.Va_k[k, b])

        # --- Solve Zone 2 ---
        for b in boundary_buses:
            for k in current_kg_and_base:
                model_z2.Va_target[k, b].set_value(z_global[k, b])
                model_z2.u_va[k, b].set_value(lam_z2[k, b])
        solver.solve(model_z2)
        
        Va_z2 = {}
        for b in boundary_buses:
            Va_z2['base', b] = pyo.value(model_z2.Va_base[b])
            for k in active_k_list:
                Va_z2[k, b] = pyo.value(model_z2.Va_k[k, b])

        # --- Global Consensus & Dual Update ---
        primal_res_sq = 0.0
        dual_res_sq = 0.0
        
        for key in z_global.keys():
            z_old = z_global[key]
            # Global consensus variable is the average of the two zones
            z_global[key] = (Va_z1[key] + Va_z2[key]) / 2.0
            
            # Dual updates
            lam_z1[key] += rho * (Va_z1[key] - z_global[key])
            lam_z2[key] += rho * (Va_z2[key] - z_global[key])
            
            # Residuals for stopping criteria
            primal_res_sq += (Va_z1[key] - z_global[key])**2 + (Va_z2[key] - z_global[key])**2
            dual_res_sq += (rho * (z_global[key] - z_old))**2
            
        true_primal_res = math.sqrt(primal_res_sq)
        true_dual_res = math.sqrt(dual_res_sq)
        scenario_residuals.append(true_primal_res)
        
        if verbose: 
            print(f"Iter {itr:02d} | Primal Res: {true_primal_res:.6f} | Dual Res: {true_dual_res:.6f} | Rho: {rho}")
        
        if true_primal_res <= tol and true_dual_res <= tol:
            break
            
        # --- Residual Balancing (Dynamic Rho) ---
        if true_primal_res > 10 * true_dual_res:
            rho = rho * 2.0
        elif true_dual_res > 10 * true_primal_res:
            rho = rho / 2.0
            
        model_z1.rho.set_value(rho)
        model_z2.rho.set_value(rho)
                
    time_ml_total = time.time() - start_ml

    # Extract final variables securely
    pg_ml_pu = {}
    for m in [model_z1, model_z2]:
        for i in m.Gens:
            val = pyo.value(m.Pg_base[i], exception=False)
            pg_ml_pu[int(i)] = val if val is not None else 0.0

    return pg_ml_pu, time_ml_total, ml_iters, scenario_residuals