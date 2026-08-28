import time
import torch
import numpy as np

# Import GNN tools
from gnn_erdos import create_pyg_dataset
# Import the exact CCGA solver from your baseline
from dcopf_model import run_ccga_algorithm

def run_ml_ccga_scenario(case, zonal_data, load_vector, PTDF_matrix, gnn_z1, gnn_z2, device, baseMVA, verbose=False):
    """Callable function that runs the GNN-Accelerated CCGA pipeline."""
    start_ml = time.time()
    
    bus_list = sorted(case['bus']['bus_i'].tolist())
    zone1_buses = zonal_data['zone1']['bus']['bus_i'].tolist()
    boundary_buses = sorted(list(zonal_data['boundary_buses']))
    global_kg = zonal_data['global_Kg']
    num_boundaries, num_global_kg = len(boundary_buses), len(global_kg)

    scenario_row_pu = np.array([load_vector[b] for b in bus_list]) / baseMVA
    load_z1 = scenario_row_pu[:len(zone1_buses)].reshape(1, -1)
    load_z2 = scenario_row_pu[len(zone1_buses):].reshape(1, -1)

    graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0].to(device)
    graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0].to(device)
    
    # =========================================================
    # 1. GNN INFERENCE (The Integer Oracle)
    # =========================================================
    with torch.no_grad():
        va_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
        zk_dummy = torch.zeros(1, num_global_kg, device=device)
        
        _, _, zk_prob_z1 = gnn_z1(graph_z1, va_dummy, va_dummy, zk_dummy, zk_dummy)
        _, _, zk_prob_z2 = gnn_z2(graph_z2, va_dummy, va_dummy, zk_dummy, zk_dummy)
        
        zk_hard_z1 = (zk_prob_z1 > 0.5).int().squeeze().cpu().numpy()
        zk_hard_z2 = (zk_prob_z2 > 0.5).int().squeeze().cpu().numpy()

    predicted_active_k = []
    for k_idx, k in enumerate(global_kg):
        z1_val = zk_hard_z1.item() if zk_hard_z1.size == 1 else zk_hard_z1[k_idx]
        z2_val = zk_hard_z2.item() if zk_hard_z2.size == 1 else zk_hard_z2[k_idx]
        
        # If either zone's GNN flags the contingency as active, trust it
        if z1_val == 1 or z2_val == 1: 
            predicted_active_k.append(k)

    if verbose:
        print(f"      -> GNN Oracle predicted {len(predicted_active_k)} active contingencies.")

    # =========================================================
    # 2. WARM-STARTED CCGA EXECUTION
    # =========================================================
    # Pass the predicted active contingencies directly into the CCGA solver
    pg_ml_pu, status, ccga_iters, final_S = run_ccga_algorithm(
        case['bus'], case['gen'], case['branch'], case['gencost'], 
        load_vector, PTDF_matrix, 
        initial_active_S=predicted_active_k
    )
    
    time_ml_total = time.time() - start_ml

    return pg_ml_pu, time_ml_total, ccga_iters, predicted_active_k