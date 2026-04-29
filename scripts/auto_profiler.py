# Used To Collect Roofline Data
#   python auto_profiler.py -s 100 --max_new_tokens 1 --min_batch_power 10 --max_batch_power 10

import subprocess
import argparse
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import csv
import pandas as pd
import datetime
import metrics_helper

from pathlib import Path
sys.path.append('..')
import logging
from utils import CustomThread
curr_date=datetime.datetime.today().strftime('%m-%d-%Y')
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, filename=f'../outputs/logs/{curr_date}.log', filemode='a')
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
    "-m",
    "--model_name",
    default="ridger/MMfreeLM-2.7B",
    help="sets the model name"
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
    '--metrics', 
    choices=['all', 'jetson'],
    default="all")


args = parser.parse_args()

ncu_path = subprocess.check_output(["which", "ncu"]).decode('ascii').strip()
logger.debug(f"Extracted ncu path: {ncu_path}")
os.chdir('../')

metrics_string = None
if(args.metrics == "all"):
    metrics_string = metrics_helper.all_metrics()
elif(args.metrics == "jetson"):
    metrics_string = metrics_helper.jetson_metrics()

def create_report_name(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    model_name = model_name.replace("/", "-")
    return  os.getcwd() + "/outputs/ncu_runs/autoProfilerFor" + model_name + "batch" + str(bs) + "newTokens" + str(new_tokens) + "sequence" + str(seq_len)
    

def run_ncu_profile(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    report_name = create_report_name(bs, new_tokens, seq_len, model_name=model_name)
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
        "python", os.getcwd() + "/quiet_run.py",
        "-b", str(bs),
        "-s", str(seq_len),
        "-n", str(new_tokens),
        "-i", "1", 
        "--model_name", model_name
    ]
    logger.debug(f"running command {' '.join(benchmark_command)}")
    # subprocess.run(benchmark_command, check=True, stdout=subprocess.DEVNULL)
    # subprocess.run(benchmark_command, check=True)
    
def flatten_kernels(df, bs, new_tokens, seq_len):
    # Conversion factors to a base unit (bytes, seconds, instructions)
    unit_conversions = {
        "byte": 1e-3,
        "Kbyte": 1,
        "Mbyte": 1e3,
        "Gbyte": 1e6,
        "us":   1,
        "ms":   1e3,
        "s":    1e6,
        "ns":   1e-3,
        "inst": 1,
    }

    # Canonical names for the base units
    canonical_unit = {
        "byte": "Kbyte", "Kbyte": "Kbyte", "Mbyte": "Kbyte", "Gbyte": "Kbyte",
        "us": "us", "ms": "us", "ns": "us", "s": "us",
        "inst": "inst",
    }

    df["Metric Value"] = df["Metric Value"].astype(float)
    df["Metric Value"] = df.apply(
        lambda r: r["Metric Value"] * unit_conversions.get(r["Metric Unit"], 1), axis=1
    )
    df["Metric Name"] = df.apply(
        lambda r: r["Metric Name"] + f' ({canonical_unit.get(r["Metric Unit"], r["Metric Unit"])})',
        axis=1
    )

    kernel_id_cols = ["ID", "Kernel Name", "Block Size", "Grid Size", "Device", "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"]

    df_flat = df.pivot_table(
        index=kernel_id_cols,
        columns="Metric Name",
        values="Metric Value",
        aggfunc="first"
    ).reset_index()
    
    logger.info(list(df_flat.columns.values))


    df_flat.columns.name = None
    double_precision_flops = (df_flat["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum (inst)"] + 
             df_flat["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum (inst)"]+
             df_flat["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum (inst)"])
    single_precision_flops =(df_flat["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum (inst)"]+
             df_flat["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum (inst)"]+
             df_flat["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum (inst)"])
    half_precision_flops = (
             df_flat["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum (inst)"]+
             df_flat["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum (inst)"]+
             df_flat["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum (inst)"])
    

    df_flat["Single Precision GFLOP/s"] = ((df_flat['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                    df_flat['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                    (df_flat['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed (inst/cycle)']*2)
            ) * df_flat['smsp__cycles_elapsed.avg.per_second (Ghz)'])
    df_flat["Half Precision GFLOP/s"] = ((df_flat['smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                df_flat['smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                (df_flat['smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed (inst/cycle)']*2)
        ) * df_flat['smsp__cycles_elapsed.avg.per_second (Ghz)'])
    df_flat["Double Precision GFLOP/s"] = ((df_flat['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                df_flat['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed (inst/cycle)'] +
                (df_flat['smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed (inst/cycle)']*2)
        ) * df_flat['smsp__cycles_elapsed.avg.per_second (Ghz)'])

    tensor_inst_rate= "smsp__inst_executed_pipe_tensor_op_hmma.sum.per_second (inst/ns)" 
    tensor_flop_count = "smsp__ops_path_tensor_src_fp16_dst_fp32.sum (nan)" 
    tensor_flop_rate =  "smsp__ops_path_tensor_src_fp16_dst_fp32.sum.per_second (nan)"
    if tensor_flop_rate not in df_flat: 
        tensor_flop_rate = 'smsp__ops_path_tensor_src_fp16_dst_fp32.sum.per_second (1/s)'

    df_flat["Half Precision Matrix Multiply and Accumulate Instructions (Inst/s)"] = df_flat[tensor_inst_rate]/1e-9
    tensor_flops = df_flat[tensor_flop_count]
    df_flat["Tensor Math Ops (16bit to 32 bit) (Billion Per Second)"] = df_flat[tensor_flop_rate] / 1e9
    
    df_flat["(Double Precision) Compute Intensity"] = double_precision_flops / (df_flat["dram__bytes.sum (Kbyte)"] * 1e3)
    df_flat["(Single Precision) Compute Intensity"] = single_precision_flops / (df_flat["dram__bytes.sum (Kbyte)"] * 1e3)
    df_flat["(Half Precision) Compute Intensity"] = half_precision_flops / (df_flat["dram__bytes.sum (Kbyte)"] * 1e3)
    df_flat["(Tensor Cores) Compute Intensity"] = tensor_flops / (df_flat["dram__bytes.sum (Kbyte)"] * 1e3)
    df_flat["Workload"] = f"Batch{bs}, NewTokens: {new_tokens} Sequence Length: {seq_len}"
    return df_flat

def extract_flops(df): 
    double_precision_flops = (
        df["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum (inst)"].sum() + 
        df["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum (inst)"].sum() +
        df["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum (inst)"].sum()
    )
    single_precision_flops = (
        df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum (inst)"].sum() +
        df["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum (inst)"].sum() +
        df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum (inst)"].sum()
    )

    half_precision_flops = (
        df["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum (inst)"].sum() +
        df["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum (inst)"].sum() +
        df["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum (inst)"].sum()
    )

    tensor_count = df["smsp__ops_path_tensor_src_fp16_dst_fp32.sum (nan)"].sum() 
    
    return double_precision_flops, single_precision_flops, half_precision_flops, tensor_count


def extract_dram_usage(df):
    # total_kilo_bytes = 0
    # total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Kbyte")]["Metric Value"].astype(float).sum()
    # total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Mbyte")]["Metric Value"].astype(float).sum() * 1e3
    # total_kilo_bytes+= df[(df["Metric Name"] == "dram__bytes.sum") & (df["Metric Unit"] == "Gbyte")]["Metric Value"].astype(float).sum() * 1e6
    return df["dram__bytes.sum (Kbyte)"].sum()

def extract_run_time(df):
    # run_time_us = 0 
    # run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "us")]["Metric Value"].astype(float).sum()
    # run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "ms")]["Metric Value"].astype(float).sum() * 1e3
    # run_time_us += df[(df["Metric Name"] == "gpu__time_duration.sum") & (df["Metric Unit"] == "s")]["Metric Value"].astype(float).sum() * 1e6
    return df["gpu__time_duration.sum (us)"].sum()

def get_metrics_from_data_frame(df, fraction_of_memory_from_weights):
    total_kilo_bytes = 0
    double_precision_count = 0 
    single_precision_count = 0 
    half_precision_count = 0 
    tensor_count = 0
    run_time_us = 0

    total_kilo_bytes += extract_dram_usage(df)
    double_precision_count, single_precision_count, half_precision_count, tensor_count = extract_flops(df)
    run_time_us += extract_run_time(df)

    results = {}
    results["Gigabytes Accessed"] = total_kilo_bytes / 1e6
    results["(Effective) Gigabytes Accessed"] = ((total_kilo_bytes*fraction_of_memory_from_weights)/8 + ((1-fraction_of_memory_from_weights)*total_kilo_bytes)) / 1e6

    results["Double Precision FLOPs"] = double_precision_count
    results["Single Precision FLOPs"] = single_precision_count
    results["Half Precision FLOPs"] = half_precision_count
    results["Tensor FLOPs"] = tensor_count
    results["Run Time (s)"] = run_time_us / 1e6
    results["(NCU) Total GFLOPs"] = (double_precision_count + single_precision_count + half_precision_count + tensor_count) / 1e9
    results["GFLOPs/s"] = results["(NCU) Total GFLOPs"] / results["Run Time (s)"]
    results["Compute Intensity"] = results["(NCU) Total GFLOPs"] / results["Gigabytes Accessed"]
    results["(Effective) Compute Intensity"] = results["(NCU) Total GFLOPs"] / results["(Effective) Gigabytes Accessed"]
    results["(Double Precision) Compute Intensity"] = (double_precision_count/1e9) / results["Gigabytes Accessed"]
    results["(Single Precision) Compute Intensity"] = (single_precision_count/1e9) / results["Gigabytes Accessed"]
    results["(Half Precision) Compute Intensity"] = (half_precision_count/1e9) / results["Gigabytes Accessed"]
    results["(Tensor) Compute Intensity"] = (tensor_count/1e9) / results["Gigabytes Accessed"]

    return results

def extract_dataframe_from_ncu_files_via_csv(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    logger.info("attempting to extract data using a csv")
    report_name = create_report_name(bs, new_tokens, seq_len, model_name=model_name)
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
    return df


def estimate_fraction_of_memory_from_weights(bs, new_tokens, seq_len):
    weight_floats = (2560*2560*4 + 13824*2560 + 2560*6912)*32 + 32000*2560
    activation_floats = (2560*bs*seq_len*5 + 6912*bs*seq_len)*32 + (bs*seq_len*2560)
    output_floats = (2560*bs*seq_len*5 + 13824*bs*seq_len)*32 + (bs*seq_len*32000)
    return weight_floats/(weight_floats+ activation_floats + output_floats)

def create_rows(bs, new_tokens, seq_len, model_name='ridger/MMfreeLM-2.7B'):
    df = extract_dataframe_from_ncu_files_via_csv(bs, new_tokens, seq_len, model_name=model_name)
    df.head(n=10000).to_csv(f"outputs/csvs/unflattened_kernels-{curr_date}.csv")
    df = flatten_kernels(df, bs, new_tokens, seq_len)
    df.head(n=10000).to_csv(f"outputs/csvs/flattened_kernels-{curr_date}.csv")
    fraction_of_memory_from_weights = estimate_fraction_of_memory_from_weights(bs, new_tokens, seq_len)
    full_workload_row = get_metrics_from_data_frame(df)
    full_workload_row['Workload'] = f'2.7B end to end with batch size: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'
    extract_additional_workload_data(df, full_workload_row['Workload'])

    has_attention = df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitAttentionForward')
    has_mlp = df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitMLP')
    has_linear = df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('linearFunction')
    first_group_of_attention_kernels_df = get_continous_group_of_kernals(df, has_attention, 0)
    first_group_of_mlp_kernels_df = get_continous_group_of_kernals(df, has_mlp, 0)
    linear_kernels_df = get_continous_group_of_kernals(df, has_linear, 0)

    first_atte_region_row = get_metrics_from_data_frame(first_group_of_attention_kernels_df, fraction_of_memory_from_weights)
    first_atte_region_row['Workload'] = f'2.7B first HGRNBitAttention region: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    first_mlp_region_row = get_metrics_from_data_frame(first_group_of_mlp_kernels_df, fraction_of_memory_from_weights)
    first_mlp_region_row['Workload'] = f'2.7B first HGRNBitMLP region: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    linear_region_row = get_metrics_from_data_frame(linear_kernels_df, fraction_of_memory_from_weights)
    linear_region_row['Workload'] = f'2.7B first linearFunction region: {bs}, tokens generated: {new_tokens}, sequence length: {seq_len}'

    first_pair = pd.concat([first_group_of_attention_kernels_df, first_group_of_mlp_kernels_df])
    first_pair.to_csv(f"outputs/csvs/first_layer_kernels-{curr_date}.csv")

    logger.info(f"full workload row generated: {full_workload_row}")
    return [full_workload_row, first_atte_region_row, first_mlp_region_row, linear_region_row]

def extract_additional_workload_data(df, workload_str):
    double_precision_count, single_precision_count, half_precision_count, tensor_count = extract_flops(df)
    flop_count = double_precision_count +  single_precision_count +  half_precision_count +  tensor_count
    run_time_us = extract_run_time(df)
    dram_kbytes_accessed = extract_dram_usage(df)

    atten_df = df[df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitAttentionForward')]
    atten_double_precision_count, atten_single_precision_count, atten_half_precision_count, atten_tensor_count = extract_flops(atten_df)
    atten_flop_count = atten_double_precision_count +  atten_single_precision_count +  atten_half_precision_count +  atten_tensor_count
    atten_run_time_us = extract_run_time(atten_df)
    atten_dram_kbytes_accessed = extract_dram_usage(atten_df)

    mlp_df = df[df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('HGRNBitMLP')]
    mlp_double_precision_count, mlp_single_precision_count, mlp_half_precision_count, mlp_tensor_count = extract_flops(mlp_df)
    mlp_flop_count = mlp_double_precision_count +  mlp_single_precision_count +  mlp_half_precision_count +  mlp_tensor_count
    mlp_run_time_us = extract_run_time(mlp_df)
    mlp_dram_kbytes_accessed = extract_dram_usage(mlp_df)

    linear_df = df[df["thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"].str.contains('linearFunction')]
    linear_double_precision_count, linear_single_precision_count, linear_half_precision_count, linear_tensor_count = extract_flops(linear_df)
    linear_flop_count = linear_double_precision_count +  linear_single_precision_count +  linear_half_precision_count +  linear_tensor_count
    linear_run_time_us = extract_run_time(linear_df)
    linear_dram_kbytes_accessed = extract_dram_usage(linear_df)


    with open(f"outputs/txt/additional_workload_info{curr_date}.txt", "w") as f:
        f.write(f"{workload_str}\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{int(double_precision_count):,d} 64 bit floating point operations\n")
        f.write(f"{int(single_precision_count):,d} 32 bit floating point operations\n")
        f.write(f"{int(half_precision_count):,d} 16 bit floating point operations\n")
        f.write(f"{int(tensor_count):,d} tensor floating point operations\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(double_precision_count/flop_count)*100}% of the FLOPs are 64 bit floating point operations\n")
        f.write(f"{(single_precision_count/flop_count)*100}% of the FLOPs are 32 bit floating point operations\n")
        f.write(f"{(half_precision_count/flop_count)*100}% of the FLOPs are 16 bit floating point operations\n")
        f.write(f"{(tensor_count/flop_count)*100}% of the FLOPs are tensor floating point operations\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(atten_flop_count/flop_count)*100}% of the FLOPs are from kernels marked with HGRNBitAttentionForward\n")
        f.write(f"{(mlp_flop_count/flop_count)*100}% of the FLOPs are from kernels marked with HGRNBitMLP\n")
        f.write(f"{(linear_flop_count/flop_count)*100}% of the FLOPs are from kernels marked with Linear\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(atten_dram_kbytes_accessed/dram_kbytes_accessed)*100}% of the dram bytes accessed are from kernels marked with HGRNBitAttentionForward\n")
        f.write(f"{(mlp_dram_kbytes_accessed/dram_kbytes_accessed)*100}% of the dram bytes accessed are from kernels marked with HGRNBitMLP\n")
        f.write(f"{(linear_dram_kbytes_accessed/dram_kbytes_accessed)*100}% of the dram bytes accessed are from kernels marked with Linear\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(atten_run_time_us/run_time_us)*100}% of the runtime is from kernels marked with HGRNBitAttentionForward\n")
        f.write(f"{(mlp_run_time_us/run_time_us)*100}% of the runtime is from kernels marked with  HGRNBitMLP\n")
        f.write(f"{(linear_run_time_us/run_time_us)*100}% of the runtime is from kernels marked with Linear\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(atten_double_precision_count/atten_flop_count)*100}% of the FLOPs in HGRNBitAttentionForward are 64 bit floating point operations\n")
        f.write(f"{(atten_single_precision_count/atten_flop_count)*100}% of the FLOPs in HGRNBitAttentionForward are 32 bit floating point operations\n")
        f.write(f"{(atten_half_precision_count/atten_flop_count)*100}% of the FLOPs in HGRNBitAttentionForward are 16 bit floating point operations\n")
        f.write(f"{(atten_tensor_count/atten_flop_count)*100}% of the FLOPs in HGRNBitAttentionForward are tensor floating point operations\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(mlp_double_precision_count/mlp_flop_count)*100}% of the FLOPs in HGRNBitMLP are 64 bit floating point operations\n")
        f.write(f"{(mlp_single_precision_count/mlp_flop_count)*100}% of the FLOPs in HGRNBitMLP are 32 bit floating point operations\n")
        f.write(f"{(mlp_half_precision_count/mlp_flop_count)*100}% of the FLOPs in HGRNBitMLP are 16 bit floating point operations\n")
        f.write(f"{(mlp_tensor_count/mlp_flop_count)*100}% of the FLOPs in HGRNBitMLP are tensor floating point operations\n")
        f.write(f"==============================================================================================\n")
        f.write(f"{(linear_double_precision_count/linear_flop_count)*100}% of the FLOPs in Linear are 64 bit floating point operations\n")
        f.write(f"{(linear_single_precision_count/linear_flop_count)*100}% of the FLOPs in Linear are 32 bit floating point operations\n")
        f.write(f"{(linear_half_precision_count/linear_flop_count)*100}% of the FLOPs in Linear are 16 bit floating point operations\n")
        f.write(f"{(linear_tensor_count/linear_flop_count)*100}% of the FLOPs in Linear are tensor floating point operations\n")

    # fraction of 

def get_continous_group_of_kernals(df, condition, index):
    changes = condition.ne(condition.shift())
    groups = changes.cumsum()
    first_true_group = groups[condition].iloc[index]
    first_region = df[condition & (groups == first_true_group)]
    return first_region
    
def log_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        logger.info(df)


def main():
    logger.info("Extracting Roofline Data")
    from datetime import datetime
    filename = f'outputs/csvs/roofline_data{curr_date}.csv'
    sequence_length = int(args.sequence_length)
    max_new_tokens = int(args.max_new_tokens)
    min_batch_power = int(args.min_batch_power)
    max_batch_power = int(args.max_batch_power)
    model_name = args.model_name
    first_row = True
    start = time.perf_counter()
    threads = []
    with open(filename, 'w') as csvfile:
        for batch_power in range(min_batch_power, max_batch_power+1):
            batch_size = 2**batch_power
            run_ncu_profile(batch_size, max_new_tokens, sequence_length, model_name=model_name)
            thread = CustomThread(target=create_rows, args=(batch_size, max_new_tokens, sequence_length, model_name))
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
