import os 
import glob 
from torch_geometric.data import Data, InMemoryDataset
import deepgate as dg 
import torch
import random
import time
import threading
import copy
import numpy as np 

from utils.utils import run_command, hash_arr
from parse_graph import parse_sdf
import utils.circuit_utils as circuit_utils

raw_dir = 'LCM_output_flatten'
genlib_path = './raw_data/genlib/sky130.csv'

save_graph_npz = 'LCM_dataset/graphs.npz'
ff_keys = 'dfrtp'

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
if __name__ == '__main__':
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    sdf_list = glob.glob(os.path.join(raw_dir, '*/*.sdf'))
    tot_time = 0
    graphs = {}
    
    for sdf_k, sdf_path in enumerate(sdf_list):
        if 'ALSU-Arithmetic-Logic-Shift-Unit' not in sdf_path:
            continue
        
        print('\n===============================================')
        print(sdf_path)
        start_time = time.time()
        circuit_name = sdf_path.split('/')[-2]
        if not os.path.exists(sdf_path):
            print('[INFO] Skip: {:}, No SDF'.format(circuit_name))
            continue
        
        # Parse SDF 
        x_data, edge_index, fanin_list, fanout_list = parse_sdf(sdf_path)
        
        # Remove FF, convert seq to comb 
        x_data, edge_index, fanin_list, fanout_list = circuit_utils.seq_to_comb(x_data, fanin_list, ff_keys=ff_keys)
        
        # Check empty
        if len(x_data) < 10 or len(edge_index) < 10:
            print('[INFO] Skip empty design: {:}'.format(circuit_name))
            continue
        
        # Statistics
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            circuit_name, sdf_k, len(sdf_list), len(x_data), 
            tot_time, tot_time / ((sdf_k + 1) / len(sdf_list)) - tot_time, 
            len(graphs)
        ))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Circuit features
        truth_table = []
        is_PI = []
        for idx in range(len(x_data)):
            cell_type = x_data[idx][1]
            if cell_type == 'PI':
                truth_table.append([0 for _ in range(64)])
                is_PI.append(1)
            else:
                tt = cell_dict[cell_type]['tt']
                while len(tt) < 64:
                    tt = tt + tt
                tt = [int(tt[i]) for i in range(64)]
                truth_table.append(tt)
                is_PI.append(0)
                
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, len(x_data))
        level_list = circuit_utils.get_level(x_data, fanin_list, fanout_list)
        
        # Save graph features 
        graph = OrderedData()
        graph.x = torch.tensor(truth_table, dtype=torch.float)
        graph.edge_index = edge_index
        graph.is_pi = is_PI
        graph.name = circuit_name
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
        
        # DeepGate2 labels
        prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.cpp_simulation(
            x_data, fanin_list, fanout_list, level_list, cell_dict, 
            no_patterns=15000
        )
        if len(prob) == 0:
            continue
        for idx in range(len(x_data)):
            if len(fanin_list[idx]) == 0 and len(fanout_list[idx]) == 0:
                prob[idx] = 0.5
        graph.connect_pair_index = con_index.T
        graph.connect_label = con_label
        assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
        if len(tt_pair_index) == 0:
            tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            tt_pair_index = tt_pair_index.t().contiguous()
        graph.prob = prob
        graph.tt_pair_index = tt_pair_index
        graph.tt_sim = tt_sim
        
        # Statistics 
        graph.no_nodes = len(x_data)
        graph.no_edges = len(edge_index[0])
        end_time = time.time()
        tot_time += end_time - start_time
        
        # Save 
        g = {}
        for key in graph.keys():
            if key == 'name' or key == 'batch' or key == 'ptr':
                continue
            if torch.is_tensor(graph[key]):
                g[key] = graph[key].cpu().numpy()
            else:
                g[key] = graph[key]
        graphs[circuit_name] = copy.deepcopy(g)
        
    np.savez_compressed(save_graph_npz, circuits=graphs)
    print(save_graph_npz)
    print(len(graphs))