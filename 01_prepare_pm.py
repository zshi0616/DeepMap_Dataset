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
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils import run_command, hash_arr
from parse_graph import parse_sdf
import utils.circuit_utils as circuit_utils

raw_dir = '/Users/zhengyuanshi/studio/DeepCircuitX_v2/data/sub_v_dcout'
# genlib_path = 'genlib/sky130.csv'
genlib_path = '/Users/zhengyuanshi/studio/DeepCircuitX_v2/genlib/sky130.csv'

save_graph_npz = 'npz/iccad_dc_pm.npz'
ff_keys = ['dfr', 'dfb', 'dfx', 'dfs', 'dlx', 'dlr', 'einvn']
thread_num = 4

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
def process_single(sdf_path, cell_dict):
    try:
        design_name = sdf_path.split('/')[-2]
        module_name = sdf_path.split('/')[-1].replace('.sdf', '')
        circuit_name = design_name + '_' + module_name
        # Parse SDF 
        x_data, edge_index, fanin_list, fanout_list = parse_sdf(sdf_path)
        
        # Remove FF, convert seq to comb 
        x_data, edge_index, fanin_list, fanout_list = circuit_utils.seq_to_comb(x_data, fanin_list, ff_keys=ff_keys)
        
        # Check empty
        if len(x_data) < 10 or len(edge_index) < 10:
            print('[INFO] Skip empty design: {:}'.format(circuit_name))
            raise ValueError('Empty design')
        
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
        
        # Check loop
        loop = circuit_utils.find_loop(fanout_list)
        if len(loop) > 0:
            print('Loop: ', loop)
            raise ValueError('Loop')
                
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
        prob, tt_pair_index, tt_dis, con_index, con_label = circuit_utils.cpp_simulation(
            x_data, fanin_list, fanout_list, level_list, cell_dict, 
            no_patterns=15000
        )
        if len(prob) == 0:
            raise ValueError('Simulation failed')
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
        graph.tt_dis = tt_dis
        
        # Statistics 
        graph.no_nodes = len(x_data)
        graph.no_edges = len(edge_index[0])
        
        # Save 
        g = {}
        for key in graph.keys:
            if key == 'name' or key == 'batch' or key == 'ptr':
                continue
            if torch.is_tensor(graph[key]):
                g[key] = graph[key].cpu().numpy()
            else:
                g[key] = graph[key]
        print('[INFO] Circuit: {:}, Nodes: {:}, Edges: {:}'.format(
            circuit_name, graph.no_nodes, graph.no_edges))
        return {'graph': g, 'name': circuit_name}
    except Exception as e:
        print('[ERROR] Circuit: {:}, Error: {:}'.format(circuit_name, e))
        return None
        
if __name__ == '__main__':
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    sdf_list = glob.glob(os.path.join(raw_dir, '*/*.sdf'))
    graphs = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        feature_to_run = {
            executor.submit(process_single, sdf_path, cell_dict): sdf_path for sdf_path in sdf_list
        }
        completed = 0
        
        for future in as_completed(feature_to_run):
            completed += 1
            result = future.result()
            if result is not None:
                graphs[result['name']] = result['graph']
            
            print('Completed: {:} / {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
                completed, len(sdf_list), 
                time.time() - start_time, 
                (time.time() - start_time) / completed * (len(sdf_list) - completed), 
                len(graphs)
            ))

    np.savez_compressed(save_graph_npz, circuits=graphs)
    print(save_graph_npz)
    print(len(graphs))