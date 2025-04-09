import os 
import glob 
from torch_geometric.data import Data, InMemoryDataset
import deepgate as dg 
import torch
import random
import time
import threading
import copy
import argparse
import numpy as np 

from utils.utils import run_command, hash_arr
from parse_graph import parse_sdf
import utils.circuit_utils as circuit_utils
import utils.dataset_utils as dataset_utils 

def get_args():
    parser = argparse.ArgumentParser(description='Prepare DeepCell Dataset')
    parser.add_argument('--sdf_dir', type=str, default='../../data/lcm/sub_v_dcout', help='SDF directory')
    parser.add_argument('--genlib_path', type=str, default='./genlib/sky130.csv', help='Genlib path')
    parser.add_argument('--input_npz_path', type=str, default='../CircuitX/npz/train.npz', help='Read graph NPZ')
    
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=100000, help='End index')
    parser.add_argument('--output_npz_path', type=str, default='', help='NPZ path')
    parser.add_argument('--list_path', type=str, default='./list/sdf_list.txt', help='List path')
    parser.add_argument('--save_name', type=str, default='train', help='Save name')
    
    # Modes 
    parser.add_argument('--outlist', action='store_true', help='Output list of sdfs')
    args = parser.parse_args()
    
    # Output path 
    if args.output_npz_path == '':
        args.output_npz_path = './npz/{}_{:}_{:}.npz'.format(args.save_name, args.start_idx, args.end_idx)
    
    return args

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x
        
def construct_node_feature(x, num_gate_types):
    # the one-hot embedding for the gate types
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    # if node_reconv:
    #     reconv = torch.tensor(x[:, 7], dtype=torch.float).unsqueeze(1)
    #     x_torch = torch.cat([x_torch, reconv], dim=1)
    return x_torch
        
if __name__ == '__main__':
    args = get_args()
    if args.outlist:
        sdf_list = glob.glob(os.path.join(args.sdf_dir, '*/*.sdf'))
        with open(args.list_path, 'w') as f:
            for sdf_path in sdf_list:
                f.write(sdf_path + '\n')
        print('Write: {:}'.format(args.list_path))
        exit(0)
    else:
        f = open(args.list_path, 'r')
        sdf_list = f.readlines()
        f.close
        sdf_list = [x.strip() for x in sdf_list]
        no_sdfs = min(len(sdf_list), args.end_idx - args.start_idx)

    # Parse AIG 
    aig = np.load(args.input_npz_path, allow_pickle=True)['circuits'].item()
    
    # Parse stdlib
    cell_dict = circuit_utils.parse_genlib(args.genlib_path)
    tot_time = 0
    graphs = {}
    
    for sdf_k, sdf_path in enumerate(sdf_list):
        if sdf_k < args.start_idx or sdf_k >= args.end_idx:
            continue
        if not os.path.exists(sdf_path):
            print('File not found: {}'.format(sdf_path))
            continue

        # Read PM
        start_time = time.time()
        arr = sdf_path.replace('.sdf', '').split('\\')
        design_name = arr[-2]
        module_name = arr[-1]
        circuit_name = design_name + '_' + module_name
        if circuit_name not in aig:
            print('[INFO] Skip: {:}, No AIG'.format(circuit_name))
            continue

        # # Debug 
        # design_name = 'uart_programmable_rv32i'
        # module_name = 'tt_um_enieman_DW01_ash_0'
        # circuit_name = design_name + '_' + module_name
        # sdf_path = os.path.join(sdf_dir, design_name, module_name + '.sdf')

        x_data, edge_index, fanin_list, fanout_list = parse_sdf(sdf_path)
        if len(edge_index) == 0 or len(x_data) < 10:
            continue

        # Read AIG
        aig_x_data = aig[circuit_name]['x']
        aig_edge_index = aig[circuit_name]['edge_index']
        aig_prob = aig[circuit_name]['prob']
        aig_forward_index = aig[circuit_name]['forward_index']
        aig_backward_index = aig[circuit_name]['backward_index']
        aig_forward_level = aig[circuit_name]['forward_level']
        aig_backward_level = aig[circuit_name]['backward_level']
        aig_tt_pair_index = aig[circuit_name]['tt_pair_index']
        aig_tt_sim = aig[circuit_name]['tt_sim']
        
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            circuit_name, sdf_k, no_sdfs, len(x_data), 
            tot_time, tot_time / ((sdf_k + 1 - args.start_idx) / no_sdfs) - tot_time, 
            len(graphs)
        ))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if len(edge_index) == 0 or len(x_data) < 5:
            continue
        
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
        graph.aig_x = aig_x_data
        graph.aig_edge_index = aig_edge_index
        graph.aig_prob = aig_prob
        graph.aig_forward_index = aig_forward_index
        graph.aig_backward_index = aig_backward_index
        graph.aig_forward_level = aig_forward_level
        graph.aig_backward_level = aig_backward_level
        graph.aig_gate = torch.zeros((len(aig_x_data), 1), dtype=torch.float)
        graph.aig_tt_pair_index = aig_tt_pair_index
        graph.aig_tt_sim = aig_tt_sim
        for idx in range(len(aig_x_data)):
            if aig_x_data[idx][1] == 1:
                graph.aig_gate[idx] = 1
            elif aig_x_data[idx][2] == 1:
                graph.aig_gate[idx] = 2
        
        # DeepGate2 labels
        prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.cpp_simulation(
            x_data, fanin_list, fanout_list, level_list, cell_dict, 
            no_patterns=15000, head='{}_{}'.format(circuit_name, args.save_name), 
            simulator = './simulator/simulator.exe', 
            max_pairs=len(x_data) * 10
        )
        for idx in range(len(x_data)):
            if len(fanin_list[idx]) == 0 and len(fanout_list[idx]) == 0:
                prob[idx] = 0.5
        # graph.connect_pair_index = con_index.T
        # graph.connect_label = con_label
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

        if len(graphs) % 5000 == 0:
            np.savez(args.output_npz_path, circuits=graphs)
            print(args.output_npz_path)
            print(len(graphs))
        
    np.savez(args.output_npz_path, circuits=graphs)
    print(args.output_npz_path)
    print(len(graphs))