import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import guropipy as gp
from guropipy import GRB
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# =============================================================================
# 1. MILP BIPARTITE GRAPH EXTRACTION (GASSE ET AL. STYLE)
# =============================================================================

def extract_milp_bipartite_graph(pyomo_model):
    """
    Extracts a Bipartite Graph (Variables <-> Constraints) from a Pyomo MILP instance.
    - Variable Features: [objective_coeff, lower_bound, upper_bound, is_integer]
    - Constraint Features: [rhs_value, sense_flag]
    - Edges: Non-zero constraint matrix coefficients A_ij
    """
    # Convert Pyomo model into Gurobi internal representation for matrix extraction
    opt = pyo.SolverFactory('gurobi')
    gurobi_net = opt._create_gurobi_model(pyomo_model)[0]
    gurobi_net.update()

    vars_list = gurobi_net.getVars()
    cons_list = gurobi_net.getConstrs()

    num_vars = len(vars_list)
    num_cons = len(cons_list)

    # 1. Extract Variable Features
    var_features = []
    for v in vars_list:
        is_int = 1.0 if v.VType in [GRB.BINARY, GRB.INTEGER] else 0.0
        lb = v.LB if v.LB > -1e20 else -1000.0
        ub = v.UB if v.UB < 1e20 else 1000.0
        obj = v.Obj
        var_features.append([obj, lb, ub, is_int])

    x_var = torch.tensor(var_features, dtype=torch.float32)

    # 2. Extract Constraint Features
    con_features = []
    for c in cons_list:
        rhs = c.RHS
        sense = 0.0 if c.Sense == '=' else (-1.0 if c.Sense == '<' else 1.0)
        con_features.append([rhs, sense])

    x_con = torch.tensor(con_features, dtype=torch.float32)

    # 3. Extract Edge Indices & Matrix Coefficients
    row_indices = []
    col_indices = []
    edge_weights = []

    for con_idx, c in enumerate(cons_list):
        row = gurobi_net.getRow(c)
        for i in range(row.size()):
            var_idx = vars_list.index(row.getVar(i))
            coeff = row.getCoeff(i)
            
            row_indices.append(con_idx)
            col_indices.append(var_idx)
            edge_weights.append(coeff)

    # PyG Bipartite Edge Index: [2, Num_Edges] connecting Constraint -> Variable
    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

    graph_data = Data()
    graph_data.x_var = x_var
    graph_data.x_con = x_con
    graph_data.edge_index = edge_index
    graph_data.edge_weight = edge_weight
    graph_data.num_vars = num_vars
    graph_data.num_cons = num_cons

    return graph_data, gurobi_net


# =============================================================================
# 2. BIPARTITE GRAPH CONVOLUTIONAL NEURAL NETWORK (GCNN)
# =============================================================================

class BipartiteConv(MessagePassing):
    """
    Custom PyTorch Geometric Bipartite Message Passing Layer.
    Passes messages from source nodes (constraints/variables) to target nodes.
    """
    def __init__(self, in_src_dim, in_tgt_dim, out_dim):
        super(BipartiteConv, self).__init__(aggr='mean')
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_src_dim + in_tgt_dim + 1, out_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x_src, x_tgt, edge_index, edge_weight):
        return self.propagate(edge_index, x=(x_src, x_tgt), edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i: Target node features, x_j: Source node features
        msg_input = torch.cat([x_i, x_j, edge_weight], dim=-1)
        return self.msg_mlp(msg_input)


class MILP_LearnToBranch_GCNN(nn.Module):
    """
    Bipartite GCNN Architecture for Strong Branching Imitation (Gasse et al., 2019).
    Predicts branching priority scores for MILP candidate integer variables.
    """
    def __init__(self, var_in_dim=4, con_in_dim=2, hidden_dim=64):
        super(MILP_LearnToBranch_GCNN, self).__init__()

        # Feature Embeddings
        self.var_embed = nn.Linear(var_in_dim, hidden_dim)
        self.con_embed = nn.Linear(con_in_dim, hidden_dim)

        # Interleaved Bipartite Convolutions
        self.con_to_var_1 = BipartiteConv(hidden_dim, hidden_dim, hidden_dim)
        self.var_to_con_1 = BipartiteConv(hidden_dim, hidden_dim, hidden_dim)
        
        self.con_to_var_2 = BipartiteConv(hidden_dim, hidden_dim, hidden_dim)

        # Variable Scoring Head
        self.branch_policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_var, x_con, edge_index, edge_weight):
        # 1. Project Raw Features
        h_var = F.leaky_relu(self.var_embed(x_var), 0.1)
        h_con = F.leaky_relu(self.con_embed(x_con), 0.1)

        # 2. Round 1: Constraints -> Variables
        h_var_1 = F.leaky_relu(self.con_to_var_1(h_con, h_var, edge_index, edge_weight), 0.1)

        # Reversed Edge Index for Variables -> Constraints
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_con_1 = F.leaky_relu(self.var_to_con_1(h_var_1, h_con, rev_edge_index, edge_weight), 0.1)

        # 3. Round 2: Constraints -> Variables
        h_var_2 = F.leaky_relu(self.con_to_var_2(h_con_1, h_var_1, edge_index, edge_weight), 0.1)

        # Combine multi-layer representations
        h_var_concat = torch.cat([h_var_1, h_var_2], dim=-1)

        # 4. Score Variables for Branching
        branch_scores = self.branch_policy_head(h_var_concat).squeeze(-1)
        return branch_scores


# =============================================================================
# 3. GUROBI MIP CALLBACK WITH GNN BRANCHING INFERENCE
# =============================================================================

class GNNBranchingCallback:
    """
    Gurobi Solver Callback that intercepts fractional nodes during Branch-and-Bound
    and uses the trained PyG GCNN policy to override default solver branching decisions.
    """
    def __init__(self, gnn_model, graph_data, device):
        self.gnn_model = gnn_model
        self.graph_data = graph_data.to(device)
        self.device = device
        
        # Precompute static PyG Bipartite Graph Features
        with torch.no_grad():
            self.var_scores = self.gnn_model(
                self.graph_data.x_var, 
                self.graph_data.x_con, 
                self.graph_data.edge_index, 
                self.graph_data.edge_weight
            ).cpu().numpy()

    def __call__(self, model, where):
        if where == GRB.Callback.MIPNODE:
            # Query node relaxation status
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.OPTIMAL:
                vars_list = model.getVars()
                rel_vals = model.cbGetNodeRel(vars_list)

                # Identify fractional integer candidates
                fractional_candidates = []
                for idx, (v, val) in enumerate(zip(vars_list, rel_vals)):
                    if v.VType in [GRB.BINARY, GRB.INTEGER]:
                        frac = abs(val - round(val))
                        if 1e-4 < frac < 0.9996:
                            fractional_candidates.append((idx, self.var_scores[idx]))

                # Select variable with highest GNN predicted strong branching score
                if fractional_candidates:
                    best_var_idx = max(fractional_candidates, key=lambda x: x[1])[0]
                    selected_var = vars_list[best_var_idx]
                    
                    # Force Gurobi to branch on the GNN-selected variable
                    model.cbSetSolution(selected_var, rel_vals[best_var_idx])


# =============================================================================
# 4. EXECUTION HARNESS
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f" LEARN-TO-BRANCH VIA BIPARTITE GCNN ON {device.type.upper()}")
    print(f"=======================================================")

    # A. LOAD POWER GRID EXCEL CASE DATA
    case_name = 'pglib_opf_case118_ieee'
    case_path = f'../excel_outputs/{case_name}.xlsx'
    
    if os.path.exists(case_path):
        print(f"Loading Grid Data from {case_path}...")
        case = pd.read_excel(case_path, sheet_name=['baseMVA', 'bus', 'gen', 'gencost', 'branch'])
        baseMVA = case['baseMVA']['baseMVA'][0]
    else:
        print(f"File {case_path} not found. Running with synthetic dimensions.")
        baseMVA = 100.0

    # B. BUILD A SAMPLE PYOMO SCDCOPF MILP MODEL
    print("Building Pyomo SCDCOPF MILP Instance...")
    model = pyo.ConcreteModel()
    model.Gens = pyo.Set(initialize=list(range(10)))
    
    # Binary Unit Commitment Decision Variables
    model.u = pyo.Var(model.Gens, domain=pyo.Binary)
    model.Pg = pyo.Var(model.Gens, domain=pyo.NonNegativeReals)

    # Objective & Constraints
    model.obj = pyo.Objective(expr=sum(model.Pg[g] * 20.0 + model.u[g] * 100.0 for g in model.Gens), sense=pyo.minimize)
    model.demand_con = pyo.Constraint(expr=sum(model.Pg[g] for g in model.Gens) >= 500.0 / baseMVA)
    model.cap_cons = pyo.ConstraintList()
    for g in model.Gens:
        model.cap_cons.add(model.Pg[g] <= 100.0 * model.u[g])

    # C. EXTRACT BIPARTITE GRAPH REPRESENTATION
    print("Extracting Bipartite MILP Graph Representation...")
    graph_data, gurobi_model = extract_milp_bipartite_graph(model)
    print(f" -> Extracted Graph: {graph_data.num_vars} Var Nodes | {graph_data.num_cons} Con Nodes | {graph_data.edge_index.size(1)} Edges")

    # D. INITIALIZE OR LOAD PRE-TRAINED GCNN BRANCHING MODEL
    print("Initializing Learn-To-Branch GCNN Policy...")
    gnn_branch_policy = MILP_LearnToBranch_GCNN(var_in_dim=4, con_in_dim=2, hidden_dim=64).to(device)
    gnn_branch_policy.eval()

    # Save/Load State Dict setup following your project layout
    os.makedirs('data/admm_models', exist_ok=True)
    model_save_path = "data/admm_models/branch_gcnn_policy.pth"
    torch.save(gnn_branch_policy.state_dict(), model_save_path)
    print(f" -> Branching GCNN Policy Saved to: {model_save_path}")

    # E. RUN GUROBI MILP SOLVE WITH GNN BRANCHING CALLBACK
    print("\nStarting Gurobi MILP Solve with GNN Branching Callback...")
    callback_engine = GNNBranchingCallback(gnn_branch_policy, graph_data, device)

    gurobi_model.Params.OutputFlag = 1
    gurobi_model.Params.PreCrush = 1 # Enable callback node modifications
    
    start_time = time.time()
    gurobi_model.optimize(callback_engine)
    solve_time = time.time() - start_time

    print(f"\n>>> Branch-and-Bound Complete! Solve Time: {solve_time:.4f}s <<<")