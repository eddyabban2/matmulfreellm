import os
import re
import csv
path = "paper_profiling"
dir_list = os.listdir(path)
dir_list = [file for file in dir_list if file.endswith(".txt")]
filename = "paper_profiling/summary.csv"
print("Files and directories in '", path, "' :")

print(dir_list)

first_row = True
with open(filename, 'w') as csvfile:
    for file in dir_list: 
        pattern = re.compile(
        r'^(?P<model>.+?)'
        r'batch(?P<batch>\d+)'
        r'newTokens(?P<output_len>\d+)'
        r'sequence(?P<input_len>\d+)'
        r'\.txt$'
        )
        m = pattern.search(file)
        if m is None:
            print("regex failed, there is nothing I can do")
            exit()
        values = m.groupdict()
        values['model'] = "microsoft/bitnet-b1.58-2B-4T" if values['model'] == 'bitnet' else values['model']
        values['model'] = "100B MatMulFreeLM" if values['model'] == 'scaled_mmfree' else values['model']
        values['model'] = "Scaled Up Bitnet" if values['model'] == 'scaled_bitnet' else values['model']
        row = {'model': values['model'], 'batch size': values['batch'], "Sequence Length": values["input_len"], "Output Length": values["output_len"]}

        relevant_labels = ['workload', 'decode', 'prefill', 'ternary matmul', 'unpack weights', 'BitLinear Forward', 'activation quantization', 'post quantization processing', 'Fused Bit Linear', "LayerNormLinearQuantFn is rms norm", "applying scale"]
        for label in relevant_labels: 
            row[label + " runtime(ns)"] = 0
            row[label + " intstances"] = 0
        with open("paper_profiling/" + file, "r") as fp:
            for line in fp:
                if "PushPop" in line: 
                    label = line.split(":")[1].strip()
                    if label in relevant_labels: 
                        list_line = line.split()
                        runtime = int(list_line[1].replace(",", ""))
                        intstances = int(list_line[2].replace(",", ""))
                        row[label + " runtime(ns)"] = runtime
                        row[label + " intstances"] = intstances
                        print(line)
                        print(f'\tlabel: [{label}]')
                        print(f"\truntime: {runtime}")
                        print(f"\tinstances: {intstances}")
        if(first_row):
            csvwriter = csv.DictWriter(csvfile, row.keys())
            csvwriter.writeheader()
            first_row = False
        csvwriter.writerow(row)

# with open("example.txt", "r", encoding="utf-8") as file:
#     content = file.read()
#     print(content)