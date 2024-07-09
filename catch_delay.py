import networkx as nx
import re
import csv
import os 
import glob

save_csv_dir = './output_delay/'
sdf_dir = './raw_data/sdf/'

def parse_delay(input_filename, csv_filename):
    
    G = nx.DiGraph()
    
    def find_max_number_in_line(line):  
        
        numbers = re.findall(r'\b\d*\.\d+\b', line)  
        
        if numbers:
            return max(float(num) for num in numbers)  


    with open(input_filename, 'r') as file:
        cell_blocks = []
        in_cell_block = False
        current_block = ''
        for line in file:
            stripped_line = line.strip()
            words = stripped_line.split()
            
            if words[0] == '(INTERCONNECT':
                
                node_name_1 = words[1]
                node_name_2 = words[2]
                # max_num = find_max_number_in_line(line)
                max_num_r = find_max_number_in_line(words[3])
                max_num_f = find_max_number_in_line(words[4])

                if node_name_1 not in G:
                    G.add_node(node_name_1)
                if node_name_2 not in G:
                    G.add_node(node_name_2)
                if max_num_r > max_num_f:
                    chose_rf = True
                else:
                    chose_rf = False
                G.add_edge(node_name_1,node_name_2,weight_r = max_num_r,weight_f=max_num_f,chose = chose_rf)
            if words[0] == '(CELL':
                in_cell_block = True
                current_block = stripped_line
            elif words[0] == ')':
                in_cell_block = False
                if current_block != '':
                    cell_blocks.append(current_block)
                    current_block = ''
            elif in_cell_block:
                current_block += '\n' + stripped_line

    for index, block in enumerate(cell_blocks, start=1):  
        lines = block.strip().split('\n')
        words = lines[2].split()
        if len(words)>=2:
            cell_ID = words[1][:-1]
            index = 5
            while index <= len(lines)-1:
                words_delay = lines[index].split()

                if words_delay[1] == '(posedge' or words_delay[1] =='(negedge':
                    pin = words_delay[2][:-1]
                    pout = words_delay[3]
                    max_num_r = find_max_number_in_line(words_delay[4])
                    max_num_f = find_max_number_in_line(words_delay[5])
                else:
                    pin = words_delay[1]
                    pout = words_delay[2]
                    max_num_r = find_max_number_in_line(words_delay[3])
                    max_num_f = find_max_number_in_line(words_delay[4])

                pin_name = cell_ID + '/' + pin
                pout_name = cell_ID + '/' + pout

                if pin_name not in G:
                    G.add_node(pin_name)
                if pout_name not in G:
                    G.add_node(pout_name)

                if G.has_edge(pin_name, pout_name): 
                    if max_num_r > G[pin_name][pout_name]['weight_r']:
                        G[pin_name][pout_name]['weight_r'] = max_num_r
                    if max_num_f > G[pin_name][pout_name]['weight_f']:
                        G[pin_name][pout_name]['weight_f'] = max_num_f
                    if G[pin_name][pout_name]['weight_r'] > G[pin_name][pout_name]['weight_f']:
                        chose_rf = True
                    else:
                        chose_rf = False
                    G[pin_name][pout_name]['chose'] = chose_rf
                    for pred in G.predecessors(pout_name):
                        qw = G[pred][pout_name]['chose']
                        for pred_pred in G.predecessors(pred):
                            G[pred_pred][pred]['chose'] = qw
                else:
                    if max_num_r > max_num_f:
                        chose_rf = True
                    else:
                        chose_rf = False
                    G.add_edge(pin_name,pout_name,weight_r = max_num_r,weight_f = max_num_f,chose = chose_rf)        
                    for pred in G.predecessors(pout_name):
                        qw = G[pred][pout_name]['chose']
                        for pred_pred in G.predecessors(pred):
                            G[pred_pred][pred]['chose'] = qw
                index = index + 1

    max_path_weights = {node: 0 for node in G.nodes()}   
    
    for node in G.nodes():  
        max_weight = 0  
        for pred in G.predecessors(node):
            if G[pred][node]['chose']:
                weight = G[pred][node]['weight_r']
            else:
                weight = G[pred][node]['weight_f']
                
            max_weight = max(max_weight, max_path_weights[pred] + weight)
        max_path_weights[node] = max_weight

    with open(csv_filename,'w') as outfile:
        outfile.write('Index' + ',' + 'name' + ',' + 'delay' + '\n')
        i = 0
        for node, weight in max_path_weights.items(): 
            if node[-1] in ['X','Y']:
                outfile.write(f"{i},{node[:-2]},{weight:.4f}\n")
                i = i+1
                # print(f"{node}: {weight:.3f}")
            
            
if __name__ == '__main__':
    if not os.path.exists(save_csv_dir):
        os.mkdir(save_csv_dir)
        
    for sdf_path in glob.glob(os.path.join(sdf_dir, '*.sdf')):
        sdf_name = os.path.basename(sdf_path).split('.')[0]
        csv_path = os.path.join(save_csv_dir, sdf_name + '.csv')
        parse_delay(sdf_path, csv_path)
        print('Save to: {:}'.format(csv_path))
    
    
