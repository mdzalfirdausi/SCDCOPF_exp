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
def build_admm_master(bus_df, gen_df, branch_df, cost_df, load_vector, active_Kg, active_Ke, baseMVA=100.0):
    model = pyo.ConcreteModel()
    
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    model.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    model.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    model.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    
    model.Active_Kg = pyo.Set(initialize=active_Kg)
    model.Active_Ke = pyo.Set(initialize=active_Ke)
    
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
    
    # Cost-to-go estimators for both contingency types
    model.t_kg = pyo.Var(model.Active_Kg, within=pyo.NonNegativeReals)
    model.t_ke = pyo.Var(model.Active_Ke, within=pyo.NonNegativeReals)

    # --- Objective ---
    def obj_rule(m):
        nominal_cost = sum(c1[i] * (m.Pg[i] * baseMVA) for i in m.Gens)
        future_cost_g = sum(m.t_kg[k] for k in m.Active_Kg) if len(active_Kg) > 0 else 0.0
        future_cost_e = sum(m.t_ke[e] for e in m.Active_Ke) if len(active_Ke) > 0 else 0.0
        return nominal_cost + future_cost_g + future_cost_e
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

    model.pmax_dict = pmax 
    return model

def add_al_cut_to_master(master, iter_idx, cont_id, cont_type, P_val, g_bar, mu, beta):
    """Adds the AL Cut for either a generator or a line contingency."""
    b = pyo.Block()
    b.Gens = pyo.Set(initialize=master.Gens)
    
    b.z = pyo.Var(b.Gens, domain=pyo.Binary)
    b.yp = pyo.Var(b.Gens, within=pyo.NonNegativeReals)
    b.ym = pyo.Var(b.Gens, within=pyo.NonNegativeReals)

    b.diff_eq = pyo.Constraint(b.Gens, rule=lambda b, i: b.yp[i] - b.ym[i] == master.Pg[i] - g_bar[i])
    b.bigm_p_eq = pyo.Constraint(b.Gens, rule=lambda b, i: b.yp[i] <= master.pmax_dict[i] * b.z[i])
    b.bigm_m_eq = pyo.Constraint(b.Gens, rule=lambda b, i: b.ym[i] <= master.pmax_dict[i] * (1 - b.z[i]))

    if cont_type == 'gen':
        b.cut_eq = pyo.Constraint(rule=lambda b: master.t_kg[cont_id] >= P_val - sum(mu[i] * (master.Pg[i] - g_bar[i]) for i in b.Gens) - beta * sum(b.yp[i] + b.ym[i] for i in b.Gens))
    else:
        b.cut_eq = pyo.Constraint(rule=lambda b: master.t_ke[cont_id] >= P_val - sum(mu[i] * (master.Pg[i] - g_bar[i]) for i in b.Gens) - beta * sum(b.yp[i] + b.ym[i] for i in b.Gens))

    master.add_component(f"AL_Cut_{cont_type}_Iter{iter_idx}_Cont{cont_id}", b)


# =====================================================================
# 2. ADMM SUBPROBLEMS (Generators & Lines)
# =====================================================================
def solve_generator_subproblem(k_cont, bus_df, gen_df, branch_df, load_vector, g_bar, mu, beta, baseMVA=100.0):
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
    for _, row in gen_df.iterrows(): bus_gens[row['bus_i']].append(row['gen_ID'])
    lines_from = {b: [] for b in sub.Buses}
    lines_to = {b: [] for b in sub.Buses}
    branch_ends = {}
    for _, row in branch_df.iterrows():
        lines_from[row['bus_i']].append(row['line_ID'])
        lines_to[row['bus_j']].append(row['line_ID'])
        branch_ends[row['line_ID']] = (row['bus_i'], row['bus_j'])

    sub.Pg_prov = pyo.Var(sub.Gens, within=pyo.NonNegativeReals) 
    sub.Pgs = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)     
    sub.Theta_s = pyo.Var(sub.Buses, bounds=(-np.pi, np.pi))
    sub.Pf_s = pyo.Var(sub.Branches)
    
    sub.n_s = pyo.Var(bounds=(0, 1))
    sub.x_s = pyo.Var(sub.Gens, domain=pyo.Binary)

    sub.d_plus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    sub.d_minus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    sub.abs_diff_eq = pyo.Constraint(sub.Gens, rule=lambda m, i: m.d_plus[i] - m.d_minus[i] == m.Pg_prov[i] - g_bar[i])

    sub.v_plus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.v_minus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.sl_line = pyo.Var(sub.Branches, within=pyo.NonNegativeReals)

    M_eta = 1500.0
    def sub_obj_rule(m):
        penalty = M_eta * sum(m.v_plus[b] + m.v_minus[b] for b in m.Buses) + M_eta * sum(m.sl_line[l] for l in m.Branches)
        al_term = sum(mu[i] * (m.Pg_prov[i] - g_bar[i]) + beta * (m.d_plus[i] + m.d_minus[i]) for i in m.Gens)
        return penalty + al_term
    sub.obj = pyo.Objective(rule=sub_obj_rule, sense=pyo.minimize)

    sub.failed_gen_eq = pyo.Constraint(expr=sub.Pgs[k_cont] == 0)
    sub.global_demand_eq = pyo.Constraint(rule=lambda m: sum(m.Pgs[i] for i in m.Gens) == sum(load_vector[b] / baseMVA for b in m.Buses))

    sub.apr_eq1 = pyo.Constraint(sub.Gens, rule=lambda m, i: pyo.Constraint.Skip if i == k_cont else m.Pgs[i] - m.Pg_prov[i] - m.n_s * gamma[i] * g_hat[i] <= pmax[i] * (1 - m.x_s[i]))
    sub.apr_eq2 = pyo.Constraint(sub.Gens, rule=lambda m, i: pyo.Constraint.Skip if i == k_cont else m.Pgs[i] - m.Pg_prov[i] - m.n_s * gamma[i] * g_hat[i] >= -pmax[i] * (1 - m.x_s[i]))
    sub.apr_eq3 = pyo.Constraint(sub.Gens, rule=lambda m, i: pyo.Constraint.Skip if i == k_cont else m.Pg_prov[i] + m.n_s * gamma[i] * g_hat[i] >= pmax[i] * (1 - m.x_s[i]))
    sub.apr_eq4 = pyo.Constraint(sub.Gens, rule=lambda m, i: pyo.Constraint.Skip if i == k_cont else m.Pgs[i] >= pmax[i] * (1 - m.x_s[i]))

    sub.flow_s_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: m.Pf_s[l] == (m.Theta_s[branch_ends[l][0]] - m.Theta_s[branch_ends[l][1]]) / x[l])
    sub.limit_s_upper_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pf_s[l] - m.sl_line[l] <= rateA[l])
    sub.limit_s_lower_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 else m.Pf_s[l] + m.sl_line[l] >= -rateA[l])

    def balance_s_rule(m, b):
        return sum(m.Pgs[g] for g in bus_gens[b]) + m.v_plus[b] - m.v_minus[b] - (load_vector[b] / baseMVA) == sum(m.Pf_s[l] for l in lines_from[b]) - sum(m.Pf_s[l] for l in lines_to[b])
    sub.balance_s_eq = pyo.Constraint(sub.Buses, rule=balance_s_rule)
    sub.ref_bus_s = pyo.Constraint(expr=sub.Theta_s[ref_bus_id] == 0)

    solver = pyo.SolverFactory('gurobi_direct')
    solver.solve(sub, tee=False)
    return pyo.value(sub.obj), {i: pyo.value(sub.Pg_prov[i]) for i in sub.Gens}

def solve_line_contingency_subproblem(e_cont, bus_df, gen_df, branch_df, load_vector, g_bar, mu, beta, baseMVA=100.0):
    sub = pyo.ConcreteModel()
    ref_bus_id = bus_df.loc[bus_df['type'] == 3, 'bus_i'].values[0]
    
    sub.Buses = pyo.Set(initialize=bus_df['bus_i'].tolist())
    sub.Gens = pyo.Set(initialize=gen_df['gen_ID'].tolist())
    sub.Branches = pyo.Set(initialize=branch_df['line_ID'].tolist())
    
    x = dict(zip(branch_df['line_ID'], branch_df['x']))
    rateA = dict(zip(branch_df['line_ID'], branch_df['rateA'] / baseMVA))

    bus_gens = {b: [] for b in sub.Buses}
    for _, row in gen_df.iterrows(): bus_gens[row['bus_i']].append(row['gen_ID'])
    lines_from = {b: [] for b in sub.Buses}
    lines_to = {b: [] for b in sub.Buses}
    branch_ends = {}
    for _, row in branch_df.iterrows():
        lines_from[row['bus_i']].append(row['line_ID'])
        lines_to[row['bus_j']].append(row['line_ID'])
        branch_ends[row['line_ID']] = (row['bus_i'], row['bus_j'])

    sub.Pg_prov = pyo.Var(sub.Gens, within=pyo.NonNegativeReals) 
    sub.Theta_s = pyo.Var(sub.Buses, bounds=(-np.pi, np.pi))
    sub.Pf_s = pyo.Var(sub.Branches)
    
    sub.d_plus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    sub.d_minus = pyo.Var(sub.Gens, within=pyo.NonNegativeReals)
    sub.abs_diff_eq = pyo.Constraint(sub.Gens, rule=lambda m, i: m.d_plus[i] - m.d_minus[i] == m.Pg_prov[i] - g_bar[i])

    sub.v_plus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.v_minus = pyo.Var(sub.Buses, within=pyo.NonNegativeReals)
    sub.sl_line = pyo.Var(sub.Branches, within=pyo.NonNegativeReals)

    M_eta = 1500.0
    def sub_obj_rule(m):
        penalty = M_eta * sum(m.v_plus[b] + m.v_minus[b] for b in m.Buses) + M_eta * sum(m.sl_line[l] for l in m.Branches)
        al_term = sum(mu[i] * (m.Pg_prov[i] - g_bar[i]) + beta * (m.d_plus[i] + m.d_minus[i]) for i in m.Gens)
        return penalty + al_term
    sub.obj = pyo.Objective(rule=sub_obj_rule, sense=pyo.minimize)

    # Line Contingency Specifics
    sub.failed_line_eq = pyo.Constraint(expr=sub.Pf_s[e_cont] == 0)

    sub.flow_s_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: pyo.Constraint.Skip if l == e_cont else m.Pf_s[l] == (m.Theta_s[branch_ends[l][0]] - m.Theta_s[branch_ends[l][1]]) / x[l])
    sub.limit_s_upper_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 or l == e_cont else m.Pf_s[l] - m.sl_line[l] <= rateA[l])
    sub.limit_s_lower_eq = pyo.Constraint(sub.Branches, rule=lambda m, l: pyo.Constraint.Skip if rateA[l] == 0 or l == e_cont else m.Pf_s[l] + m.sl_line[l] >= -rateA[l])

    # No APR logic for line failures; generation is fixed to provisional dispatch
    def balance_s_rule(m, b):
        return sum(m.Pg_prov[g] for g in bus_gens[b]) + m.v_plus[b] - m.v_minus[b] - (load_vector[b] / baseMVA) == sum(m.Pf_s[l] for l in lines_from[b]) - sum(m.Pf_s[l] for l in lines_to[b])
    sub.balance_s_eq = pyo.Constraint(sub.Buses, rule=balance_s_rule)
    sub.ref_bus_s = pyo.Constraint(expr=sub.Theta_s[ref_bus_id] == 0)

    solver = pyo.SolverFactory('gurobi_direct')
    solver.solve(sub, tee=False)
    return pyo.value(sub.obj), {i: pyo.value(sub.Pg_prov[i]) for i in sub.Gens}

# =====================================================================
# 3. ALGORITHM 4.1 EXECUTION
# =====================================================================
def run_admm_al_cuts(bus, gen, branch, gencost, load_vector, active_Kg=None, active_Ke=None, max_iters=50, tolerance=1e-2):
    gens_list = gen['gen_ID'].tolist()
    
    # If sets aren't provided, default to full problem (WARNING: Very slow)
    if active_Kg is None: active_Kg = gens_list
    if active_Ke is None: active_Ke = branch['line_ID'].tolist()
        
    print("Building ADMM Master Problem...")
    master = build_admm_master(bus, gen, branch, gencost, load_vector, active_Kg, active_Ke)
    solver = pyo.SolverFactory('gurobi_direct')
    
    g_bar = {i: 0.0 for i in gens_list}
    
    # Separate Dual Tracking for Gens vs Lines
    mu_k = {k: {i: 0.0 for i in gens_list} for k in active_Kg}
    beta_k = {k: 5.0 for k in active_Kg}  
    
    mu_e = {e: {i: 0.0 for i in gens_list} for e in active_Ke}
    beta_e = {e: 5.0 for e in active_Ke}
    
    admmDualStepSize = 0.5
    prev_obj_val = 0.0
    final_iteration = 0

    for iteration in range(1, max_iters + 1):
        final_iteration = iteration
        print(f"\n--- ADMM Iteration {iteration} ---")
        
        # 1A. Evaluate active generator contingencies
        for k in active_Kg:
            P_val, local_g_prov = solve_generator_subproblem(k, bus, gen, branch, load_vector, g_bar, mu_k[k], beta_k[k])
            add_al_cut_to_master(master, iteration, k, 'gen', P_val, g_bar.copy(), mu_k[k], beta_k[k])
            for i in gens_list: mu_k[k][i] += admmDualStepSize * beta_k[k] * (local_g_prov[i] - g_bar[i])
            beta_k[k] = min(100.0, beta_k[k] * 1.1) 
            
        # 1B. Evaluate active line contingencies
        for e in active_Ke:
            P_val, local_g_prov = solve_line_contingency_subproblem(e, bus, gen, branch, load_vector, g_bar, mu_e[e], beta_e[e])
            add_al_cut_to_master(master, iteration, e, 'line', P_val, g_bar.copy(), mu_e[e], beta_e[e])
            for i in gens_list: mu_e[e][i] += admmDualStepSize * beta_e[e] * (local_g_prov[i] - g_bar[i])
            beta_e[e] = min(100.0, beta_e[e] * 1.1)
        
        # 2. Solve Master
        total_cuts = iteration * (len(active_Kg) + len(active_Ke))
        print(f"Solving Master Problem with {total_cuts} active AL Cuts...")
        results = solver.solve(master, tee=False)
        
        if results.solver.termination_condition != TerminationCondition.optimal:
            print("Master problem failed or unbounded.")
            break
            
        g_bar = {i: pyo.value(master.Pg[i]) for i in gens_list}
        obj_val = pyo.value(master.cost)
        print(f"Master Objective: ${obj_val:.2f}")

        if abs(obj_val - prev_obj_val) < tolerance and iteration > 1:
            print("Optimal AL Cut consensus reached! Objective stagnated.")
            break
        prev_obj_val = obj_val

    optimal_g = {i: g_bar[i] * 100.0 for i in gens_list}
    return optimal_g, prev_obj_val, final_iteration