"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    torchrun --nproc_per_node=2 pipelined_performance_eval.py -s 100 -m 20 -w 1.5 -l 1.5 --max_new_tokens 100 --min_batch_power 0 --max_batch_power 5
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv
from utils import generate_random_input_ids, generate_dataset_input_ids
from pipeline_mmfreelm import PipelineParallelMatMulFreeLM

parser = argparse.ArgumentParser(
    description="creates a csv file with benchmark results"
)

parser.add_argument(
    "-s", 
    "--sequence_length",
    default=5,
    help="sets the sequence length of input tokens"
)


parser.add_argument(
    "-m", 
    "--micro_batches",
    default=10,
    help="sets the number of micro batches"
)

parser.add_argument(
    "-w", 
    "--weight_multiplier",
    default=0.5,
    help="sets the number of we multiply the number of weights in each layer by"
)

parser.add_argument(
    "-l", 
    "--layers_multiplier",
    default=0.5,
    help="sets the number  we multiply the number of layers by"
)

parser.add_argument( 
    "--max_new_tokens",
    default=20,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-i", 
    "--iterations",
    default=1,
    help="Determines the number of iterations to benchmark for"
)


parser.add_argument(
    "--min_batch_power", 
    default=0,
    help="stores the minimum batch power to go up to when profiling",
)

parser.add_argument(
    "--max_batch_power", 
    default=3,
    help="stores the maximum batch power to go up to when profiling",
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()


def benchmark_generation(pipelined_model, batch_size, seq_len, num_iterations, max_new_tokens, row, num_micro_batches, rank, use_dataset_prompts=False, model_id="ridger/MMfreeLM-2.7B"):
    """Run benchmark with multiple prompts and iterations."""
    
    results = {
            'tps': [],
            'generation_time': [],
            'average_power_watts': [],
            'max_power_watts': [],
            'min_power_watts': [], 
            'total_energy_joules': [],
            'joules_per_token': []
        }
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    for gpu_idx in range(world_size):
        results[f'gpu {gpu_idx} total_energy'] = []
    micro_batches = []
    generate_input_ids = generate_dataset_input_ids if use_dataset_prompts else generate_random_input_ids
    if rank == 0: 
        for _ in range(num_micro_batches):
            inputs = generate_input_ids(model_id, batch_size, seq_len)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                })
    dist.barrier()
    devices = list(range(world_size))

    power_monitor = PowerMonitor(gpu_indices=devices)
    monitor = ZeusMonitor(gpu_indices=devices)
    window_key = "generate pipeline"

    pipelined_model.generate_pipelined(micro_batches[0:2], max_new_tokens=max_new_tokens)
    
    for _ in range(num_iterations):
        start_time = time.time()
        torch.cuda.synchronize()
        monitor.begin_window(window_key, sync_execution=True)
        pipelined_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
        mes = monitor.end_window("generate pipeline", sync_execution=True)
        torch.cuda.synchronize()
        end_time = time.time()

        generation_time = end_time - start_time
        tokens_generated = max_new_tokens*batch_size*num_micro_batches
        tps = (tokens_generated) / generation_time
        timeline = power_monitor.get_power_timeline(
            power_domain="device_instant",  # or "device_average" or "memory_average"
            start_time=start_time,
            end_time=end_time)
        for gpu_idx, data in timeline.items():
            powers = [power_watts for timestamp, power_watts in data]
            results['average_power_watts'].append(sum(powers) / len(powers))
            results['max_power_watts'].append(max(powers))
            results['min_power_watts'].append(min(powers))
        for gpu_idx in mes.gpu_energy: 
            results[f'gpu {gpu_idx} total_energy'].append(mes.gpu_energy[gpu_idx])

        results[f'total_energy_joules'].append(sum((mes.gpu_energy.values())))
        results[f'joules_per_token'].append(sum((mes.gpu_energy.values())) / tokens_generated)
        results['generation_time'].append(generation_time)
        results['tps'].append(tps)
        
    row['tokens_per_second'] = statistics.mean(results['tps'])
    row['run_time_seconds'] = statistics.mean(results["generation_time"])
    row['average_power_watts'] = statistics.mean(results['average_power_watts'])
    row['max_power_watts'] = statistics.mean(results['max_power_watts'])
    row['min_power_watts'] = statistics.mean(results['min_power_watts'])
    row['total_energy_joules'] = statistics.mean(results['total_energy_joules'])
    row['joules_per_token'] = statistics.mean(results['joules_per_token'])
    for gpu_idx in range(world_size):
        row[f'total energy from GPU: {gpu_idx}'] = statistics.mean(results[f'gpu {gpu_idx} total_energy'])


def time_to_first_token(pipelined_model, batch_size, seq_len, num_iterations, row, num_micro_batches, rank, use_dataset_prompts=False, model_id="ridger/MMfreeLM-2.7B"):
    """Estimate Time to First TOken """
    
    times_to_first_token = []
    micro_batches = []
    generate_input_ids = generate_dataset_input_ids if use_dataset_prompts else generate_random_input_ids
    if rank == 0: 
        for _ in range(num_micro_batches):
            inputs = generate_input_ids(model_id, batch_size, seq_len)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                })
    dist.barrier()

    pipelined_model.generate_pipelined(micro_batches[0:2], max_new_tokens=1)
    
    for _ in range(num_iterations):
        start_time = time.time()
        torch.cuda.synchronize()
        pipelined_model.generate_pipelined(micro_batches, max_new_tokens=1)
        torch.cuda.synchronize()
        end_time = time.time()

        times_to_first_token.append(end_time - start_time)
        
    row['time_to_first_token'] = statistics.mean(times_to_first_token)

def create_csv_data(
        sequence_length, 
        max_new_tokens, 
        iters, 
        min_batch_power, 
        max_batch_power, 
        count_micro_batches, 
        weight_multiplier, 
        layers_multiplier):

    first_row = True
    rank = int(os.environ.get("RANK", 0))
    from datetime import datetime
    current_time = datetime.now()
    filename = f"outputs/csvs/pipelined_performance_eval-{current_time:%Y-%m-%d_%H:%M:%S}.csv"
    pipelined_model = PipelineParallelMatMulFreeLM(weight_multiplier=weight_multiplier, layers_multiplier=layers_multiplier)  
    device = pipelined_model.device
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        original_hidden_layer_size = 2560
        original_num_layers = 32
        memory_usage = 0
        for device in range(world_size): 
            memory_usage += torch.cuda.memory_allocated(device=device)
        row = {
            'device': device, 
            'Hiden Layer Size': original_hidden_layer_size*weight_multiplier, 
            'Number of Layers': original_num_layers*layers_multiplier, 
            'DRAM Bytes From Model (Bytes)': memory_usage}
        for batch_power in reversed(range(min_batch_power, max_batch_power)):
            batch_size = 2**batch_power
            row['Batch Size'] = batch_size
            print(f"\tCollecting data for batch size: {batch_size}")
            print(f"\t\tRunning Benchmarks...")

            benchmark_generation(pipelined_model, batch_size, sequence_length, iters, max_new_tokens, row, count_micro_batches, rank)
            time_to_first_token(pipelined_model, batch_size, sequence_length, iters, row, count_micro_batches, rank)
            if rank == 0:
                if(first_row):
                    csvwriter = csv.DictWriter(csvfile, row.keys())
                    csvwriter.writeheader()
                    first_row = False
                csvwriter.writerow(row) 
        print(f"Data written to {filename}")

def main():
    sequence_length=int(args.sequence_length)
    max_new_tokens=int(args.max_new_tokens)
    iters=int(args.iterations)
    min_batch_power=int(args.min_batch_power)
    max_batch_power=int(args.max_batch_power)
    count_micro_batches=int(args.micro_batches)
    weight_multiplier=float(args.weight_multiplier)
    layers_multiplier=float(args.layers_multiplier)

    create_csv_data(sequence_length, max_new_tokens, iters, min_batch_power, max_batch_power, count_micro_batches, weight_multiplier, layers_multiplier)

if __name__ == "__main__":
    main()
