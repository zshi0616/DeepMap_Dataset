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
import utils.aiger_utils as aiger_utils
from parse_graph import parse_sdf
import utils.circuit_utils as circuit_utils

raw_dir = '/Users/zhengyuanshi/studio/DeepCircuitX_v2/data/sub_aig'
save_graph_npz = 'npz/iccad_dc_aig.npz'
thread_num = 8
gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
def process_single(aig_path):
    try:
        design_name = aig_path.split('/')[-2]
        module_name = aig_path.split('/')[-1].replace('.sdf', '')
        circuit_name = design_name + '_' + module_name
        tmp_aag_filename = os.path.join('./tmp', circuit_name + '.aag')
        x_data, edge_index = aiger_utils.seqaig_to_xdata(aig_path, tmp_aag_filename)
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        
        # Replace DFF as PPI and PPO
        no_ff = 0
        for idx in range(len(x_data)):
            if x_data[idx][1] == gate_to_index['DFF']:
                no_ff += 1
                x_data[idx][1] = gate_to_index['PI']
                for fanin_idx in fanin_list[idx]:
                    fanout_list[fanin_idx].remove(idx)
                fanin_list[idx] = []
        # circuit_utils.save_bench('./tmp/test.bench', x_data, fanin_list, fanout_list)
        
        # Get x_data and edge_index
        edge_index = []
        for idx in range(len(x_data)):
            for fanin_idx in fanin_list[idx]:
                edge_index.append([fanin_idx, idx])
        x_data, edge_index = circuit_utils.remove_unconnected(x_data, edge_index)
        if len(edge_index) < 100 or len(x_data) < 100:
            raise ValueError('Empty design')
        x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
        
        graph = OrderedData()
        graph.x = x_one_hot
        graph.edge_index = edge_index
        graph.name = circuit_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
        
        ################################################
        # DeepGate2 (node-level) labels
        ################################################
        prob, tt_pair_index, tt_dis, con_index, con_label = circuit_utils.prepare_dg2_labels_cpp(graph, 15000)
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
        graph.no_nodes = x_one_hot.size(0)
        graph.no_edges = edge_index.size(1)
        
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
    aig_list = glob.glob(os.path.join(raw_dir, '*/*.aig'))
    graphs = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        feature_to_run = {
            executor.submit(process_single, aig_path): aig_path for aig_path in aig_list
        }
        completed = 0
        
        for future in as_completed(feature_to_run):
            completed += 1
            result = future.result()
            if result is not None:
                graphs[result['name']] = result['graph']
            
            print('Completed: {:} / {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
                completed, len(aig_list), 
                time.time() - start_time, 
                (time.time() - start_time) / completed * (len(aig_list) - completed), 
                len(graphs)
            ))

    np.savez_compressed(save_graph_npz, circuits=graphs)
    print(save_graph_npz)
    print(len(graphs))