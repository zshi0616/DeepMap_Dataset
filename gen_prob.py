import numpy as np 
import os 
import glob 

import utils.circuit_utils as circuit_utils

genlib_path = './genlib/sky130.csv'
netlist_dir = './epfl/abc_res/sky130'

if __name__ == '__main__':
    # Parse stdlib
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    
    for netlist_path in glob.glob(os.path.join(netlist_dir, '*.v')):
        x_data, fanin_list, fanout_list, PI_index, PO_index = circuit_utils.parse_v(netlist_path)
        print('Processing: {}, # Nodes: {:}'.format(netlist_path, len(x_data)))
        
        # Circuit features 
        level_list = circuit_utils.get_level(x_data, fanin_list, fanout_list)
        print('Max Level: {:}'.format(len(level_list)))
        
        # Simulation 
        prob, tt_index, tt_sim, con_index, con_label = circuit_utils.cpp_simulation(
            x_data, fanin_list, fanout_list, level_list, PI_index, PO_index, cell_dict, 
            no_patterns=15000
        )
        print()
        
    