import time
import concurrent.futures
import pyomo.environ as pyo

# ============================================================================
# 1. Subproblem Builders
# ============================================================================

def build_x_subproblem(h_omega, prob, T, Z_max):
    """
    Builds the x-subproblem for a single scenario[cite: 1].
    Uses mutable parameters to allow fast re-solving.
    """
    m = pyo.ConcreteModel()
    
    # Mutable parameters for ADMM updates
    m.z_bar1 = pyo.Param(mutable=True, default=0.0)
    m.z_bar2 = pyo.Param(mutable=True, default=0.0)
    m.mu1 = pyo.Param(mutable=True, default=0.0)
    m.mu2 = pyo.Param(mutable=True, default=0.0)
    m.beta = pyo.Param(mutable=True, default=1.0)
    
    # Variables
    m.x1 = pyo.Var(domain=pyo.Binary)
    m.x2 = pyo.Var(domain=pyo.Binary)
    m.x3 = pyo.Var(domain=pyo.Binary)
    m.x4 = pyo.Var(domain=pyo.Binary)
    m.z1 = pyo.Var(bounds=(0, Z_max), domain=pyo.Integers)
    m.z2 = pyo.Var(bounds=(0, Z_max), domain=pyo.Integers)
    
    # L1 norm linearization for the positive penalty beta * || z - z_bar ||_1
    m.v1 = pyo.Var(bounds=(0, None))
    m.v2 = pyo.Var(bounds=(0, None))
    m.c_v1_pos = pyo.Constraint(expr=m.v1 >= m.z1 - m.z_bar1)
    m.c_v1_neg = pyo.Constraint(expr=m.v1 >= m.z_bar1 - m.z1)
    m.c_v2_pos = pyo.Constraint(expr=m.v2 >= m.z2 - m.z_bar2)
    m.c_v2_neg = pyo.Constraint(expr=m.v2 >= m.z_bar2 - m.z2)
    
    # Constraints for the scenario
    h1, h2 = h_omega
    m.con1 = pyo.Constraint(expr=2*m.x1 + 3*m.x2 + 4*m.x3 + 5*m.x4 <= h1 - (T[0][0]*m.z1 + T[0][1]*m.z2))
    m.con2 = pyo.Constraint(expr=6*m.x1 + 1*m.x2 + 3*m.x3 + 1*m.x4 <= h2 - (T[1][0]*m.z1 + T[1][1]*m.z2))
    
    # Objective: Expected cost + multiplier term + AL penalty[cite: 1]
    cost_x = prob * (-16*m.x1 - 19*m.x2 - 23*m.x3 - 28*m.x4)
    cost_mu = m.mu1 * (m.z1 - m.z_bar1) + m.mu2 * (m.z2 - m.z_bar2)
    cost_beta = m.beta * (m.v1 + m.v2)
    
    m.obj = pyo.Objective(expr=cost_x + cost_mu + cost_beta, sense=pyo.minimize)
    return m

def build_z_master(Z_max):
    """
    Builds the initial master z-subproblem without cuts[cite: 1].
    """
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var(bounds=(0, Z_max), domain=pyo.Integers)
    m.z2 = pyo.Var(bounds=(0, Z_max), domain=pyo.Integers)
    m.t = pyo.Var(bounds=(None, None))
    
    m.obj = pyo.Objective(expr=-1.5*m.z1 - 4*m.z2 + m.t, sense=pyo.minimize)
    m.cuts = pyo.ConstraintList()
    m.cut_vars = pyo.Block(pyo.Any)
    return m

def add_al_cut(m, iteration, z_bar, P_val, mu_sum, beta, num_scenarios, Z_max):
    """
    Adds a nonconvex AL cut to the master problem[cite: 1].
    """
    b = m.cut_vars[iteration] = pyo.Block()
    
    # Auxiliary variables for EXACT absolute value | z - z_bar |
    b.y_pos = pyo.Var([1, 2], bounds=(0, Z_max))
    b.y_neg = pyo.Var([1, 2], bounds=(0, Z_max))
    b.bin = pyo.Var([1, 2], domain=pyo.Binary)
    
    b.eq1 = pyo.Constraint(expr=m.z1 - z_bar[0] == b.y_pos[1] - b.y_neg[1])
    b.eq2 = pyo.Constraint(expr=m.z2 - z_bar[1] == b.y_pos[2] - b.y_neg[2])
    
    b.M1_pos = pyo.Constraint(expr=b.y_pos[1] <= Z_max * b.bin[1])
    b.M1_neg = pyo.Constraint(expr=b.y_neg[1] <= Z_max * (1 - b.bin[1]))
    
    b.M2_pos = pyo.Constraint(expr=b.y_pos[2] <= Z_max * b.bin[2])
    b.M2_neg = pyo.Constraint(expr=b.y_neg[2] <= Z_max * (1 - b.bin[2]))
    
    L1_norm = b.y_pos[1] + b.y_neg[1] + b.y_pos[2] + b.y_neg[2]
    mu_term = mu_sum[0] * (m.z1 - z_bar[0]) + mu_sum[1] * (m.z2 - z_bar[1])
    
    # The AL Cut[cite: 1]
    m.cuts.add(m.t >= P_val - mu_term - beta * num_scenarios * L1_norm)

# ============================================================================
# 2. Parallel Worker Function
# ============================================================================

def solve_single_scenario(args):
    """
    Top-level function for ProcessPoolExecutor to solve a single x-subproblem.
    """
    w, m_x, z_k, mu_w, beta = args
    
    # Update mutable parameters
    m_x.z_bar1 = z_k[0]
    m_x.z_bar2 = z_k[1]
    m_x.mu1 = mu_w[0]
    m_x.mu2 = mu_w[1]
    m_x.beta = beta
    
    # Solve
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.solve(m_x)
    
    # Extract numerical results to avoid returning the unpicklable Pyomo model
    P_val = pyo.value(m_x.obj)
    z_omega = [pyo.value(m_x.z1), pyo.value(m_x.z2)]
    x_omega = [pyo.value(m_x.x1), pyo.value(m_x.x2), pyo.value(m_x.x3), pyo.value(m_x.x4)]
    
    return w, P_val, z_omega, x_omega

# ============================================================================
# 3. Main ADMM Routine
# ============================================================================

def run_admm_investment_parallel(S=21, Z_max=5, use_T_matrix=False):
    # Setup Scenarios
    scenarios = []
    for s1 in range(1, S + 1):
        for s2 in range(1, S + 1):
            h1 = 5 + 10 * (s1 - 1) / (S - 1)
            h2 = 5 + 10 * (s2 - 1) / (S - 1)
            scenarios.append((h1, h2))
            
    num_scenarios = len(scenarios)
    prob = 1.0 / num_scenarios
    T = [[2/3, 1/3], [1/3, 2/3]] if use_T_matrix else [[1, 0], [0, 1]]
    
    # Parameters exactly as reported in the paper[cite: 1]
    rho_0 = 1.0
    gamma = 1.1
    innerADMM = 50
    admmDualStepSize = 200.0
    
    # State initialization
    z_k = [0.0, 0.0] 
    mu = {w: [0.0, 0.0] for w in range(num_scenarios)}
    beta = rho_0
    
    print(f"Initializing {num_scenarios} x-subproblems...")
    x_models = [build_x_subproblem(scenarios[w], prob, T, Z_max) for w in range(num_scenarios)]
    master_model = build_z_master(Z_max)
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    
    best_UB = float('inf')
    start_time = time.time()
    
    # ADMM Loop[cite: 1]
    for k in range(1, 100):
        P_val = 0.0
        z_omega_k = {}
        x_omega_k = {}
        
        # Parallel solve for all x-subproblems[cite: 1]
        tasks = [(w, x_models[w], z_k, mu[w], beta) for w in range(num_scenarios)]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(solve_single_scenario, tasks)
            
            # Aggregate the results from all CPU processes
            for w, P_val_w, z_omega, x_omega in results:
                P_val += P_val_w
                z_omega_k[w] = z_omega
                x_omega_k[w] = x_omega
                
        # Check Primal Feasibility (L1 Norm)[cite: 1]
        infeasibility = sum(abs(z_omega_k[w][0] - z_k[0]) + abs(z_omega_k[w][1] - z_k[1]) for w in range(num_scenarios))
        
        if infeasibility < 1e-6:
            # Candidate Upper Bound calculation[cite: 1]
            UB_candidate = -1.5*z_k[0] - 4*z_k[1] + sum(
                prob * (-16*x_omega_k[w][0] - 19*x_omega_k[w][1] - 23*x_omega_k[w][2] - 28*x_omega_k[w][3]) 
                for w in range(num_scenarios)
            )
            best_UB = min(best_UB, UB_candidate)
            
        # Generate AL cut and solve the z-subproblem[cite: 1]
        mu_sum = [sum(mu[w][0] for w in range(num_scenarios)), sum(mu[w][1] for w in range(num_scenarios))]
        add_al_cut(master_model, k, z_k, P_val, mu_sum, beta, num_scenarios, Z_max)
        
        solver.solve(master_model)
        
        next_z_k = [pyo.value(master_model.z1), pyo.value(master_model.z2)]
        LB = pyo.value(master_model.obj)
        
        # Calculate Gap[cite: 1]
        gap = (best_UB - LB) / max(1e-10, abs(best_UB)) if best_UB != float('inf') else float('inf')
        
        print(f"Iter {k:2d} | LB: {LB:8.2f} | UB: {best_UB:8.2f} | Gap: {gap:.2%} | L1-Infeas: {infeasibility:6.2f} | z: {next_z_k}")
        
        if best_UB != float('inf') and gap <= 1e-4:
            print(f"\nOptimal Solution Verified! Total Time: {time.time() - start_time:.2f} seconds")
            break
            
        # Update Dual Multipliers and Penalty Parameter[cite: 1]
        for w in range(num_scenarios):
            mu[w][0] += admmDualStepSize * beta * (z_omega_k[w][0] - next_z_k[0])
            mu[w][1] += admmDualStepSize * beta * (z_omega_k[w][1] - next_z_k[1])
            
        if k % innerADMM == 0:
            beta *= gamma
            
        z_k = next_z_k

# Required safeguard for Windows compatibility when using ProcessPoolExecutor
if __name__ == "__main__":
    # Reproduces the S=21 instance from Table 1 with parallel acceleration[cite: 1]
    run_admm_investment_parallel(S=21, Z_max=5, use_T_matrix=False)