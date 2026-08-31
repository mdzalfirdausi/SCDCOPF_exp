import torch
import numpy as np
from gnn_erdos import create_zonal_data, create_pyg_dataset

def predict_active_contingencies(case, load_vector, gnn_z1, gnn_z2, device):
    """
    Takes raw grid data and a load vector, passes it through the trained GNNs,
    and returns lists of predicted active generator and line contingencies.
    """
    baseMVA = case['baseMVA']['baseMVA'][0]
    total_buses = case['bus']['bus_i'].tolist()
    midpoint = len(total_buses) // 2
    zone1_buses, zone2_buses = total_buses[:midpoint], total_buses[midpoint:]
    
    zonal_data = create_zonal_data(case, zone1_buses, zone2_buses)
    global_kg = zonal_data['global_Kg']
    global_ke = zonal_data['global_Ke']
    
    num_boundaries = len(zonal_data['boundary_buses'])
    num_global_kg = len(global_kg)
    num_global_ke = len(global_ke)
    
    bus_list = sorted(case['bus']['bus_i'].tolist())
    scenario_row_pu = np.array([load_vector[b] for b in bus_list]) / baseMVA
    
    load_z1 = scenario_row_pu[:len(zone1_buses)].reshape(1, -1)
    load_z2 = scenario_row_pu[len(zone1_buses):].reshape(1, -1)
    
    graph_z1 = create_pyg_dataset(zonal_data['zone1'], load_z1, baseMVA)[0].to(device)
    graph_z2 = create_pyg_dataset(zonal_data['zone2'], load_z2, baseMVA)[0].to(device)
    
    graph_z1.batch = torch.zeros(graph_z1.num_nodes, dtype=torch.long, device=device)
    graph_z2.batch = torch.zeros(graph_z2.num_nodes, dtype=torch.long, device=device)
    
    gnn_z1.eval()
    gnn_z2.eval()
    
    with torch.no_grad():
        va_dummy = torch.zeros(1, num_global_kg + 1, num_boundaries, device=device)
        zk_dummy = torch.zeros(1, num_global_kg, device=device)
        ek_dummy = torch.zeros(1, num_global_ke, device=device)
        
        # The GNN returns 4 items: Pg_base, Va_local, zk_local, ek_local
        _, _, zk_prob_z1, ek_prob_z1 = gnn_z1(graph_z1, va_dummy, va_dummy, zk_dummy, zk_dummy, ek_dummy, ek_dummy)
        _, _, zk_prob_z2, ek_prob_z2 = gnn_z2(graph_z2, va_dummy, va_dummy, zk_dummy, zk_dummy, ek_dummy, ek_dummy)
        
        final_zk_prob = (zk_prob_z1 + zk_prob_z2) / 2.0
        final_ek_prob = (ek_prob_z1 + ek_prob_z2) / 2.0
        
        binary_preds_k = (final_zk_prob.squeeze() > 0.5).int().cpu().numpy()
        binary_preds_e = (final_ek_prob.squeeze() > 0.5).int().cpu().numpy()
        
    predicted_active_kg = [global_kg[i] for i, val in enumerate(binary_preds_k) if val == 1]
    predicted_active_ke = [global_ke[i] for i, val in enumerate(binary_preds_e) if val == 1]
    
    return predicted_active_kg, predicted_active_ke