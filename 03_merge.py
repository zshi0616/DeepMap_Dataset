import numpy as np 
import torch
import os
from torch_geometric.data import Data, InMemoryDataset

pm_npz_path = 'npz/iccad_dc_pm.npz'
aig_npz_path = 'npz/iccad_dc_aig.npz'
output_npz_path = 'npz/iccad_dc_merged.npz'

MAX_SIZE = -1

def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
if __name__ == '__main__':
    aigs = read_npz_file(aig_npz_path)['circuits'].item()
    pms = read_npz_file(pm_npz_path)['circuits'].item()
    graphs = {}
    
    for pm_idx, pm_name in enumerate(pms):
        aig_name = pm_name + '.aig'
        if aig_name not in aigs:
            print(f'PM {pm_name} not in AIG')
            continue
        aig = aigs[aig_name]
        aig_x_data = aig['x']
        aig_edge_index = aig['edge_index']
        aig_prob = aig['prob']
        aig_forward_index = aig['forward_index']
        aig_backward_index = aig['backward_index']
        aig_forward_level = aig['forward_level']
        aig_backward_level = aig['backward_level']
        aig_tt_pair_index = aig['tt_pair_index']
        aig_tt_dis = aig['tt_dis']
        pm = pms[pm_name]
        
        if MAX_SIZE > 0 and len(aig['x']) + len(pm['x']) > MAX_SIZE:
            print(f'Skipping {pm_name} due to size limit')
            continue
        
        graph = OrderedData()
        graph.x = pm['x']
        graph.prob = pm['prob']
        graph.edge_index = pm['edge_index']
        graph.name = pm_name
        graph.forward_level = pm['forward_level']
        graph.backward_level = pm['backward_level']
        graph.forward_index = pm['forward_index']
        graph.backward_index = pm['backward_index']
        graph.tt_dis = pm['tt_dis']
        graph.tt_pair_index = pm['tt_pair_index']
        graph.connect_pair_index = pm['connect_pair_index']
        graph.connect_label = pm['connect_label']
        graph.no_nodes = pm['no_nodes']
        graph.no_edges = pm['no_edges']
        
        # Add AIG features
        graph.aig_x = aig_x_data
        graph.aig_edge_index = aig_edge_index
        graph.aig_prob = aig_prob
        graph.aig_forward_index = aig_forward_index
        graph.aig_backward_index = aig_backward_index
        graph.aig_forward_level = aig_forward_level
        graph.aig_backward_level = aig_backward_level
        graph.aig_gate = torch.zeros((len(aig_x_data), 1), dtype=torch.float)
        graph.aig_tt_pair_index = aig_tt_pair_index
        graph.aig_tt_dis = aig_tt_dis
        for idx in range(len(aig_x_data)):
            if aig_x_data[idx][1] == 1:
                graph.aig_gate[idx] = 1
            elif aig_x_data[idx][2] == 1:
                graph.aig_gate[idx] = 2
        
        print('{} / {}'.format(pm_idx, len(pms)))
        print(f'PM: {pm_name}, AIG: {aig_name}, Nodes: {graph.no_nodes}, Edges: {graph.no_edges}')
        print()
        graphs[pm_name] = graph

    # Save merged graphs
    output_npz_path = os.path.join(output_npz_path)
    if not os.path.exists(os.path.dirname(output_npz_path)):
        os.makedirs(os.path.dirname(output_npz_path))
    np.savez_compressed(output_npz_path, circuits=graphs)
    print(f'Saved merged graphs to {output_npz_path}')
    print(f'Total circuits: {len(graphs)}')