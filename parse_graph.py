import numpy as np 
import os 
import glob 

import utils.circuit_utils as circuit_utils

sdf_dir = 'deepgate_dataset/sdf'
save_graph_npz = 'deepgate_dataset/graph_npz'

import re

def parse_sdf(file_path):
    x_data = []
    edge_index = []
    fanin_list_wpin = []
    fanin_list = []
    fanout_list = []

    # 使用正则表达式匹配INTERCONNECT内容
    interconnect_re = re.compile(r'\(INTERCONNECT (\S+)/?\S* (\S+)/?\S*')

    with open(file_path, 'r') as file:
        content = file.read()
        interconnects = interconnect_re.findall(content)

    # 提取所有出现的cell，去重
    cells_set = set()
    connections = []
    for start, end in interconnects:
        start = start.split('/')
        end = end.split('/')
        cells_set.add(start[0])
        cells_set.add(end[0])
        connections.append((start, end))

    cells_list = sorted(list(cells_set))  # 排序以确保一致性

    # 初始化 x_data, fanin_list, fanout_list
    for cell in cells_list:
        x_data.append([cell, "PI"])  # 初始时类型未知
        fanin_list_wpin.append([])
        fanin_list.append([])
        fanout_list.append([])

    # 创建从 cell 名称到索引的映射
    instance_map = {instance: index for index, instance in enumerate(cells_list)}

    # 解析 interconnects 并填充 edge_index, fanin_list, fanout_list
    for start, end in connections:
        start_index = instance_map[start[0]]
        end_index = instance_map[end[0]]
        edge_index.append([start_index, end_index])
        fanin_list_wpin[end_index].append((start_index, end[1]))
        
    # 使用正则表达式匹配CELL内容
    cell_re = re.compile(r'\(CELLTYPE "([^"]+)"\)\s+\(INSTANCE ([^\s]+)\)')
    with open(file_path, 'r') as file:
        content = file.read()
        interconnects = interconnect_re.findall(content)
        cells = cell_re.findall(content)
        
    # 不考虑顺序
    for cell_type, instance in cells:
        if instance in instance_map:
            index = instance_map[instance]
            x_data[index][1] = cell_type
            for idx, (fanin, pin) in enumerate(fanin_list_wpin[index]):
                fanin_list[index].append(fanin)
                fanout_list[fanin].append(index)
        
    # # 解析 cells 并更新 x_data 中的类型，同时记录每个cell的pin顺序
    # iopath_re = re.compile(r'\(IOPATH (.*?) \(')
    # cell_pin_order = {}
    # for cell_type, instance in cells:
    #     if instance in instance_map:
    #         index = instance_map[instance]
    #         x_data[index][1] = cell_type
            
    #         # 记录pin的顺序
    #         cell_pattern = r'\(CELL\s+\(CELLTYPE "{}"\)\s+\(INSTANCE {}\)(.*?)\)\n\)'.format(re.escape(cell_type), re.escape(instance))
    #         match = re.search(cell_pattern, content, re.DOTALL)
    #         cell_content = match.group(0) if match else None
    #         if cell_content:
    #             matches = iopath_re.findall(cell_content)
    #             raw_order = [match for match in matches]
    #         order = []
    #         for ele_idx, ele in enumerate(raw_order):
    #             if 'posedge' in ele:
    #                 pin_name = ele.split(' ')[1].split(')')[0]
    #                 if pin_name not in order:
    #                     order.append(pin_name)
    #             elif 'negedge' in ele:
    #                 pin_name = ele.split(' ')[1].split(')')[0]
    #                 if pin_name not in order:
    #                     order.append(pin_name)
    #             else:
    #                 order.append(ele.split(' ')[0])
            
    #         # Reorder: # TODO: Change here
    #         assert len(order) == len(fanin_list_wpin[index])
            
    #         fanin_list[index] = [None] * len(order)
    #         for input_pin in order:
    #             for idx, (fanin, pin) in enumerate(fanin_list_wpin[index]):
    #                 if pin == input_pin:
    #                     fanin_list[index][idx] = fanin
    #                     fanout_list[fanin].append(index)


    return x_data, edge_index, fanin_list, fanout_list

if __name__ == '__main__':
    for sdf_path in glob.glob(os.path.join(sdf_dir, '*.sdf')):
        circuit_name = os.path.basename(sdf_path).split('.')[0]
        print(sdf_path)
        x_data, edge_index, fanin_list, fanout_list = parse_sdf(sdf_path)
        
        # Check connection 
        no_edge = 0 
        for idx in range(len(fanin_list)):
            no_edge += len(fanin_list[idx])
        assert no_edge == len(edge_index)
        
        print()
        