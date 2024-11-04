import torch 
import os 
import numpy as np
import utils.dataset_utils as dataset_utils

npz_list = [
    # './npz/train_011_80000_90000.npz',
    # './npz/train_01_0_100000.npz',
    # './npz/train_02_0_30000.npz',
    # './npz/train_02_30000_90000.npz',
    # './npz/train_03_0_100000.npz',
    # './npz/train_0_100.npz',
    
    './npz/split/train_000_0_1000.npz',
    './npz/split/train_10000_10000_11000.npz',
    './npz/split/train_20000_20000_21000.npz',
    './npz/split/train_30000_30000_31000.npz',
    './npz/split/train_40000_40000_41000.npz',
    './npz/split/train_50000_50000_51000.npz',
    './npz/split/train_60000_60000_60983.npz',
]

output_npz_path = './npz/train.npz'
output_dir = './npz/train'

if __name__ == '__main__':
    graphs = {}
    
    for npz_k, npz_path in enumerate(npz_list):
        print('Loading ... {}'.format(npz_path))
        data = np.load(npz_path, allow_pickle=True)['circuits'].item()
        for cir_name in data.keys():
            ckt = data[cir_name]
            graphs[cir_name] = ckt
            print('Loaded {}'.format(cir_name))
    
    
    print('Total number of Circuits: {}'.format(len(graphs)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tp = 0
    steps = 10000
    npz_k = 0
    graph_names = list(graphs.keys())
    while tp < len(graph_names):
        graphs_tmp = {}
        for i in range(tp, min(tp+steps, len(graph_names))):
            graphs_tmp[graph_names[i]] = graphs[graph_names[i]]
        output_npz_path = os.path.join(output_dir, 'train_{}_{}.npz'.format(len(graphs_tmp), npz_k))
        np.savez(output_npz_path, circuits=graphs_tmp)
        print('Saved to {}'.format(output_npz_path))
        npz_k += 1
        tp += steps

