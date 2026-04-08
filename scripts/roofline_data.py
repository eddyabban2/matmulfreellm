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
import pandas as pd

from pathlib import Path
sys.path.append('..')
import mmfreelm
import logging
from utils import CustomThread

logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, filename='roofline.log', filemode='w')
logger.setLevel(logging.DEBUG)

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

parser.add_argument(
    "--min_batch_power",
    default="0",
    help="sets the minimum batch power"
)

parser.add_argument(
    "--max_batch_power",
    default="0",
    help="sets the minimum batch power"
)
parser.add_argument(
    '-t',
    '--test',
    default=False,
    action='store_true'    
)


args = parser.parse_args()

double_precision_metrics = [ "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", 
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum", 
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum" ]
single_precision_metrics = ["sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"]
half_precision_metrics = ["sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum"]
tensor_core_metrics = ["sm__ops_path_tensor_op_hmma_pred_on.sum",
        "sm__ops_path_tensor_op_imma_pred_on.sum"]
metrics_string = ",".join(["dram__bytes.sum", "gpu__time_duration.sum"] + 
        # double_precision_metrics + 
        single_precision_metrics + 
        half_precision_metrics)
        # + tensor_core_metrics)


ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
logger.debug(f"Extracted ncu path: {ncu_path}")
os.chdir('../')


def run_ncu_profile(bs, new_tokens, seq_len):
    report_name = os.getcwd() + "/ncu_runs/roofline_data_batch"+ str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    benchmark_command = [
        ncu_path, "--nvtx",
        # "--nvtx", "--nvtx-include", "workload/HGRNBitAttentionForward/HGRNBitMLP/",
        # "--nvtx-exclude", "warmup/",
        "--nvtx-include", "workload/",
        # "--nvtx-include", "HGRNBitAttentionForward/",
        # "--nvtx-include", "HGRNBitMLP/",
        "--config-file", "off",
        "--export", report_name,
        "--force-overwrite",
        "--replay-mode", "application",
        "--app-replay-match", "name",
        # "--target-processes", "all",
        "--target-processes", "application-only",
        "--metrics", metrics_string,
        "python", "quiet_run.py",
        "-b", str(bs),
        "-s", str(seq_len),
        "-n", str(new_tokens),
        "-i", "1"
    ]
    logger.debug(f"running command {' '.join(benchmark_command)}")
    # subprocess.run(benchmark_command, check=True, stdout=subprocess.DEVNULL)
    # subprocess.run(benchmark_command, check=True)
    
def extract_data_from_ncu_files(bs, new_tokens, seq_len):
    logger.info("extracting data")
    report_name = os.getcwd() + "/ncu_runs/roofline_data_batch"+ str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    extract_data_command = [
        ncu_path, 
        "--import", report_name + ".ncu-rep",  
        "--page", "raw",
        "--metrics", metrics_string
    ]

    logger.debug(f"running command {' '.join(extract_data_command)}")
    data = subprocess.check_output(extract_data_command).decode('ascii')
    logger.debug(f"data extracted: {data[:100000]}")
    logger.debug("data extracted now parsing data")
    data = data.splitlines()
    results = {}
    total_kilo_bytes, double_precision_count, single_precision_count, half_precision_count, tensor_count, run_time_us = parse_data_from_ncu_files(data)
    results["KiloBytes Accessed"] = total_kilo_bytes
    results["Double Precision FLOPs"] = double_precision_count
    results["Single Precision FLOPs"] = single_precision_count
    results["Half Precision FLOPs"] = half_precision_count
    results["Tensor FLOPs"] = tensor_count
    results["Run Time (s)"] = run_time_us / 1e6
    results["(NCU) Total FLOPs"] = double_precision_count + single_precision_count + half_precision_count + tensor_count
    results['Model Name'] = 'ridger/MMfreeLM-2.7B'
    results['Batch Size'] = bs
    results['Tokens Generated'] = new_tokens 
    results['Input Sequence Length'] = seq_len
    logger.info(f"row generated: {results}")
    return results

def get_metrics_from_data_frame(df):
    total_kilo_bytes = 0
    double_precision_count = 0 
    single_precision_count = 0 
    half_precision_count = 0 
    tensor_count = 0
    run_time_us = 0

    total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Kbyte")]["Metric Value"].astype(float).sum()
    total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Mbyte")]["Metric Value"].astype(float).sum() * 1e3
    total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Gbyte")]["Metric Value"].astype(float).sum() * 1e6

    valid_dram_units = ["Kbyte", "Mbyte", "Gbyte"]
    invalid_dram_rows = df[(df["Metric Name"] == "dram__bytes.sum") & (~df["Metric Unit"].isin(valid_dram_units))]
    if len(invalid_dram_rows) != 0:
        logger.error("Invalid Dram rows detected")
        logger.error(invalid_dram_rows.head())
        exit()

    double_precision_count += df[df["Metric Name"].isin(double_precision_metrics)]["Metric Value"].astype(float).sum()
    single_precision_count += df[df["Metric Name"].isin(single_precision_metrics)]["Metric Value"].astype(float).sum()
    half_precision_count += df[df["Metric Name"].isin(half_precision_metrics)]["Metric Value"].astype(float).sum()
    tensor_count += df[df["Metric Name"].isin(tensor_core_metrics)]["Metric Value"].astype(float).sum()

    run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "us")]["Metric Value"].astype(float).sum()
    run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "ms")]["Metric Value"].astype(float).sum() * 1e3
    run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "s")]["Metric Value"].astype(float).sum() * 1e6

    valid_time_units = ["ms", "us"]
    invalid_time_rows = df[(df["Metric Name"] == "gpu__time_duration.sum") & (~df["Metric Unit"].isin(valid_time_units))]
    if len(invalid_time_rows) != 0:
        logger.error("Invalid time rows detected")
        logger.error(invalid_time_rows.head())
        exit()

    results = {}
    results["Gigabytes Accessed"] = total_kilo_bytes / 1e6
    results["Double Precision FLOPs"] = double_precision_count
    results["Single Precision FLOPs"] = single_precision_count
    results["Half Precision FLOPs"] = half_precision_count
    results["Tensor FLOPs"] = tensor_count
    results["Run Time (s)"] = run_time_us / 1e6
    results["(NCU) Total GFLOPs"] = (double_precision_count + single_precision_count + half_precision_count + tensor_count) / 1e9
    results["GFLOPs/s"] = results["(NCU) Total GFLOPs"] / results["Run Time (s)"]
    results["Compute Intensity"] = results["(NCU) Total GFLOPs"] / results["Gigabytes Accessed"]
    return results

def extract_data_from_ncu_files_via_csv(bs, new_tokens, seq_len):
    logger.info("attempting to extract data using a csv")
    report_name = os.getcwd() + "/ncu_runs/roofline_data_batch"+ str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    csv_name = report_name +".csv"
    extract_data_command = [
        ncu_path, 
        "--import", report_name + ".ncu-rep",  
        "--csv", 
        "--metrics", metrics_string
    ]
    logger.info(' '.join(extract_data_command))
    with open(csv_name, "w") as f:
        subprocess.run(extract_data_command, check=True, stdout=f)
    df = pd.read_csv(csv_name)
    df = df.replace(',','', regex=True)

    full_workload_row = get_metrics_from_data_frame(df)
    full_workload_row['Workload'] = f'2.7B end to end with batch size: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    has_attention = df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitAttentionForward')
    has_mlp = df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitMLP')
    first_group_of_attention_kernels_df = get_continous_group_of_kernals(df, has_attention, 0)
    first_group_of_mlp_kernels_df = get_continous_group_of_kernals(df, has_mlp, 0)

    first_atte_region_row = get_metrics_from_data_frame(first_group_of_attention_kernels_df)
    first_atte_region_row['Workload'] = f'2.7B first HGRNBitAttention region: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    first_mlp_region_row = get_metrics_from_data_frame(first_group_of_mlp_kernels_df)
    first_mlp_region_row['Workload'] = f'2.7B first HGRNBitMLP region: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    logger.info(f"full workload row generated: {full_workload_row}")
    return [full_workload_row, first_atte_region_row, first_mlp_region_row]

def get_continous_group_of_kernals(df, condition, index):
    changes = condition.ne(condition.shift())
    groups = changes.cumsum()
    first_true_group = groups[condition].iloc[index]
    first_region = df[condition & (groups == first_true_group)]
    return first_region
    
def log_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        logger.info(df)

def parse_data_from_ncu_files(data):
    dram_kernel_count = 0 
    total_kilo_bytes = 0
    double_precision_count = 0 
    single_precision_count = 0 
    half_precision_count = 0 
    tensor_count = 0
    run_time_us = 0
    flop_metrics = double_precision_metrics + single_precision_metrics + half_precision_metrics + tensor_core_metrics
    for line in data: 
        if "dram__bytes.sum" in line:
            words = line.split()
            value = float(words[2])
            if('Mbyte' in line):
                value *= 1000
            if('Gbyte' in line):
                value *= 1e6
            if("Kbyte" not in line and 'Mbyte' not in line and 'Gbyte' not in line):
                logger.error(f"warning non standard value found in line{line}")
                exit()
            total_kilo_bytes += value
        elif any(metric in line for metric in flop_metrics):
            words = line.split()
            value = int(words[2].replace("," , ""))
            if any(metric in line for metric in double_precision_metrics):
                double_precision_count += value
            elif any(metric in line for metric in single_precision_metrics):
                single_precision_count += value
            elif any(metric in line for metric in half_precision_metrics):
                half_precision_count += value
            elif any(metric in line for metric in tensor_core_metrics):
                tensor_count += value
        elif "gpu__time_duration.sum" in line:
            value = float(words[2])
            if('ms' == words[1]):
                value *= 1e3
            if("us" not in line and 'ms' not in line):
                logger.error(f"warning non standard value time value in line: {line}")
                exit()
            run_time_us += value
    return total_kilo_bytes, double_precision_count, single_precision_count, half_precision_count, tensor_count, run_time_us

# def get_data(bs, new_tokens, seq_len):
#     row = {}
#     row['Model Name'] = 'ridger/MMfreeLM-2.7B'
#     row['Batch Size'] = bs
#     row['Tokens Generated'] = new_tokens 
#     row['Input Sequence Length'] = seq_len
#     data = get_ncu_data(bs, new_tokens, seq_len)
#     row["GigaBytes Accessed"] = data["KiloBytes Accessed"] / 1e6
#     row["Double Precision FLOPs"] = data["Double Precision FLOPs"]
#     row["Single Precision FLOPs"] = data["Single Precision FLOPs"]
#     row["Half Precision FLOPs"] = data["Half Precision FLOPs"]
#     row["Tensor FLOPs"] = data["Tensor FLOPs"] 
#     row["(NCU) Total GFLOPs"] = data["(NCU) Total FLOPs"] / 1e9
#     row["Run Time (s)"] = data["Run Time (s)"]
#     # row['(PyTorch) Total GFLOPs'] = get_flops(bs, new_tokens, seq_len) /1e9
#     row['Compute Intensity (NCU)'] = row["(NCU) Total GFLOPs"] / row['GigaBytes Accessed']
#     row["GFLOP/s"] = row["(NCU) Total GFLOPs"] / row["Run Time (s)"]
#     # row['Compute Intensity (PyTorch)'] = row["(PyTorch) Total GFLOPs"] / row['GigaBytes Accessed']
#     print(row)
#     return row

def main():
    logger.info("Extracting Roofline Data")
    from datetime import datetime
    filename =  'roofline_data.csv'
    sequence_length = int(args.sequence_length)
    max_new_tokens = int(args.max_new_tokens)
    min_batch_power = int(args.min_batch_power)
    max_batch_power = int(args.max_batch_power)
    first_row = True
    start = time.perf_counter()
    threads = []
    with open(filename, 'w') as csvfile:
        for batch_power in range(min_batch_power, max_batch_power+1):
            batch_size = 2**batch_power
            run_ncu_profile(batch_size, max_new_tokens, sequence_length)
            thread = CustomThread(target=extract_data_from_ncu_files_via_csv, args=(batch_size, max_new_tokens, sequence_length))
            threads.append(thread)
            thread.start()
        for thread in threads:
            rows = thread.join()
            for row in rows:
                if(first_row):
                    csvwriter = csv.DictWriter(csvfile, row.keys())
                    csvwriter.writeheader()
                    first_row = False
                csvwriter.writerow(row) 
    end = time.perf_counter()
    logger.info(f"Data for Roofline extracted")
    
if __name__ == "__main__":
    main()