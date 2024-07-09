import re
import csv
import os 
import glob 

save_csv_dir = './output_power/'
power_res_dir = './raw_data/power_dc_res/'

def parse_power(input_filename, csv_filename):
    with open(input_filename, 'r') as file, open(csv_filename,'w') as outfile:
        outfile.write('name' + ',' + 'internal'+ ',' + 'switching' + ',' + 'leakage' + ',' + 'total' + ',' + 'attrs' + '\n')
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith('U'):
                words = stripped_line.split()
                outfile.write(words[0] + ',' + words[1]+ ',' + words[2] + ',' + words[3] + ',' + words[4] + ',' + words[6][:-1] + '\n')

    regex = r'U(\d+)'
    sorted_rows = []

    with open(csv_filename, newline='', encoding='utf-8') as csvfile:  
        csvreader = csv.reader(csvfile)  
        header = next(csvreader)  
        for row in csvreader:  
            match = re.match(regex, row[0])  
            if match:  
                num = int(match.group(1))  
                sorted_rows.append((num, row))
            
    sorted_rows.sort(key=lambda x: x[0], reverse=True)
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:  
        csvwriter = csv.writer(csvfile)   
        csvwriter.writerow(header)
        for _, row in sorted_rows:  
            csvwriter.writerow(row)

    indexed_rows = []
    indexed_rows.append(['Index'] + header)  
    indexed_rows.append([0] + list(sorted_rows[0][1]))

    for i, (_, row) in enumerate(sorted_rows[1:], start=1):
        indexed_rows.append([i] + row) 
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerows(indexed_rows)  
    
if __name__ == '__main__':
    if not os.path.exists(save_csv_dir):
        os.mkdir(save_csv_dir)
        
    for power_res_path in glob.glob(os.path.join(power_res_dir, '*.txt')):
        power_name = os.path.basename(power_res_path).split('.')[0]
        csv_path = os.path.join(save_csv_dir, power_name + '.csv')
        parse_power(power_res_path, csv_path)
        print('Save to: {:}'.format(csv_path))
    
    
    
    