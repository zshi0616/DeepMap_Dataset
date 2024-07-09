import numpy as np 
import os 
import glob 

import utils.circuit_utils as circuit_utils

genlib_path = './raw_data/genlib/sky130.csv'
netlist_dir = './raw_data/epfl/abc_res/sky130'
save_csv_dir = './output_prob/'

if __name__ == '__main__':
    if not os.path.exists(save_csv_dir):
        os.mkdir(save_csv_dir)
        
    # Parse stdlib
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    
    for netlist_path in glob.glob(os.path.join(netlist_dir, '*.v')):
        circuit_name = os.path.basename(netlist_path).split('.')[0]
        csv_path = os.path.join(save_csv_dir, circuit_name + '.csv')
        
        # Read netlist
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
        
        # Output 
        f = open(csv_path, 'w')
        f.write('Index,Name,Prob\n')
        for i in range(len(prob)):
            f.write('{},{},{:.6f}\n'.format(i, x_data[i][0], prob[i]))
        f.close()
        print('Save to: {:}'.format(csv_path))
        
    