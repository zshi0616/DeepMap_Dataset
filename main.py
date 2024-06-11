import numpy as np 
import os 
import glob 

import utils.circuit_utils as circuit_utils

genlib_path = './genlib/asap7.csv'

if __name__ == '__main__':
    # Parse stdlib and netlist
    cell_dict = circuit_utils.parse_genlib(genlib_path)
    x_data, fanin_list, fanout_list, PI_index, PO_index = circuit_utils.parse_v('epfl/abc_res/asap7/adder.v')
    
    # Circuit features 
    level_list = circuit_utils.get_level(x_data, PI_index, fanout_list)
    
    # Simulation 
    prob, tt_index, tt_sim, con_index, con_label = circuit_utils.cpp_simulation(
        x_data, fanin_list, fanout_list, level_list, PI_index, PO_index, cell_dict, 
        no_patterns=15000
    )
    
    print()
        
    