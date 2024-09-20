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
import shutil

from utils.utils import run_command, hash_arr
from parse_graph import parse_sdf
import utils.circuit_utils as circuit_utils

raw_dir = 'LCM_output_flatten'
dst_dir = 'large_aig'

genlib_path = './raw_data/genlib/sky130.csv'

save_graph_npz = 'LCM_dataset/graphs.npz'
ff_keys = 'dfrtp'

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
        
if __name__ == '__main__':
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    aig_list = glob.glob(os.path.join(raw_dir, '*/*.aig'))
    tot_time = 0
    graphs = {}
    no_succ = 0
    no_tot = 0
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for aig_k, aig_path in enumerate(aig_list):
        print('\n===============================================')
        print(aig_path)
        no_tot += 1
        start_time = time.time()
        circuit_name = aig_path.split('/')[-2]
        if not os.path.exists(aig_path):
            continue
        if os.path.getsize(aig_path) < 1024 * 100:
            continue
        
        # Copy 
        dst_path = os.path.join(dst_dir, circuit_name + '.aig')
        shutil.copy(aig_path, dst_path)
        no_succ += 1
        
    print('{} / {}'.format(no_succ, no_tot))