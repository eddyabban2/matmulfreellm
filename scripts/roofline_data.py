# Used To Collect Roofline Data
# example run: 
#   python roofline_data.py -s 161 --max_new_tokens 338	

import subprocess
import argparse
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import csv
sys.path.append('..')
import mmfreelm
from bench_utils import generate_random_input_ids

parser = argparse.ArgumentParser(
    description="Collects data for use in a roofline graph"
)

parser.add_argument(
    "-s",
    "--sequence_length",
    default="1",
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "--max_new_tokens",
    default="1",
    help="sets the number of new tokens to be generated"
)

args = parser.parse_args()


def get_dram_kbytes_used(bs, new_tokens, seq_len):
    ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
    print(f"Extracted ncu path: {ncu_path}")

    report_name = "/home/eabban/eddy_matmulfreellm/ncu_runs/roofline_data_batch"+ str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    benchmark_command = [
        ncu_path, 
        "--nvtx", "--nvtx-include", "workload/",
        "--config-file", "off",
        "--export", report_name,
        "--force-overwrite",
        # "--target-processes", "all",
        "--target-processes", "application-only",
        "--metrics", ",".join([
            # Memory metrics
            "dram__bytes.sum"
            # collecting this data with nsight compute is too intensive 
            # #F64 opeartions
            # "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", 
            # "sm__sass_thread_inst_executed_op_dfma_pred_on.sum", 
            # "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
            # # FP32 operations
            # "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
            # "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
            # "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",  # counts as 2 FLOPS
            # # FP16 operations
            # "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
            # "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
            # "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",  # counts as 2 FLOPS
            # # Tensor Core operations (if using matmul/transformer ops)
            # "sm__ops_path_tensor_op_hmma_pred_on.sum",           # FP16 tensor core
            # "sm__ops_path_tensor_op_imma_pred_on.sum",           # INT8 tensor core
        ]),
        "/home/eabban/eddy_matmulfreellm/quiet_run.py",
        "-b", str(bs),
        "-s", str(seq_len),
        "-n", str(new_tokens),
        "-i", "1"
    ]
    print(f"running command {' '.join(benchmark_command)}")
    subprocess.run(benchmark_command, check=True, stdout=subprocess.DEVNULL)
    # subprocess.run(benchmark_command, check=True)
    print("command finished now extracting data")

    extract_data_command = [
        ncu_path, 
        "--import", report_name + ".ncu-rep",  
        "--page", "raw",
        "--metrics", "dram__bytes.sum"
    ]

    print(f"running command {' '.join(extract_data_command)}")
    data = subprocess.check_output(extract_data_command).decode('ascii')
    print("data extracted now parsing data")
    data = data.splitlines()
    kernel_count = 0 
    total = 0
    for line in data: 
        if "dram__bytes.sum" in line:
            kernel_count += 1
            words = line.split()
            value = float(words[2])
            if('Mbyte' in line):
                value *= 1000
            if('Gbyte' in line):
                value *= 1e6
            if("Kbyte" not in line and 'Mbyte' not in line and 'Gbyte' not in line):
                print(f"warning non standard value found in line{line}")
                exit()
            total += value
    print(f"Kernal Count: {kernel_count}")
    print(f"Total {total:,.2f} Kbytes accessed")
    return total
def get_flops(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, bs, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True
    ) as prof:
        _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                do_sample=True,
                top_p=0.4,
                temperature=0.6
            )
    events = prof.key_averages()
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    float_ops = sum(e.flops for e in events)
    return float_ops

def get_data(bs, new_tokens, seq_len):
    row = {}
    row['Model Name'] = 'ridger/MMfreeLM-2.7B'
    row['Batch Size'] = bs
    row['Tokens Generated'] = new_tokens 
    row['Input Sequence Length'] = seq_len
    row['GigaBytes Accessed'] = get_dram_kbytes_used(bs, new_tokens, seq_len) / 1e6
    # time.sleep(120)
    row['GFLOPs'] = gigaFlops = get_flops(bs, new_tokens, seq_len)/ 1e9
    row['Compute Intensity'] = row['GFLOPs'] / row['GigaBytes Accessed']
    print(row)
    return row

def main():
    print("Extracting Roofline Data")
    from datetime import datetime
    filename =  'roofline_data.csv'.format(date=datetime.now())
    sequence_length = int(args.sequence_length)
    max_new_tokens = int(args.max_new_tokens)
    min_batch_power = 7
    max_batch_power = 10
    first_row = True
    with open(filename, 'w') as csvfile:
        for batch_power in range(min_batch_power, max_batch_power+1):
            batch_size = 2**batch_power
            print(batch_size)
            row = get_data(batch_size, max_new_tokens, sequence_length)
            if(first_row):
                csvwriter = csv.DictWriter(csvfile, row.keys())
                csvwriter.writeheader()
                first_row = False
            csvwriter.writerow(row) 


    

if __name__ == "__main__":
    main()