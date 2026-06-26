"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    torchrun --nproc_per_node=2 pipelined_performance_eval.py -s 100 -m 5 -w 1.5 -l 1.5 --max_new_tokens 10 --min_batch_power 0 --max_batch_power 2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv
from utils import generate_random_input_ids, generate_dataset_input_ids
from pipeline_mmfreelm import PipelineParallelMatMulFreeLM
import gc

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
    "-v", 
    "--vocab_multiplier",
    default=1.5,
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

parser.add_argument(
    "--collect_power_data",
    action='store_true',
    default=False,
    help="sets whether we should collect power data"
)

parser.add_argument(
    "--print_csv",
    action='store_true',
    default=False,
    help="prints csv after creating data"
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()


def benchmark_generation(pipelined_model, batch_size, seq_len, num_iterations, max_new_tokens, row, num_micro_batches, rank, use_dataset_prompts=False, model_id="ridger/MMfreeLM-2.7B", collect_power_data=False):
    """Run benchmark with multiple prompts and iterations."""
    
    results = {
            'tps': [],
            'generation_time': [],
            'average_power_watts': [],
            'max_power_watts': [],
            'min_power_watts': [], 
            'total_energy_joules': []
        }
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    if collect_power_data:
        results['joules_per_token'] = []
        results['average_power_watts'] = []
        results['max_power_watts'] = []
        results['min_power_watts'] = [] 
        results['total_energy_joules'] = []
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
    if collect_power_data:
        power_monitor = PowerMonitor(gpu_indices=devices)
        monitor = ZeusMonitor(gpu_indices=devices)
        window_key = "generate pipeline"

    print("running warmup")
    pipelined_model.generate_pipelined(micro_batches[0:world_size], max_new_tokens=1)
    print("finished warmup")
    row["Estimated Memory Usage Per GPU Memory Usage (GB)"] = (torch.cuda.memory_allocated())/(1024**3)
    row["Estimated Total Memory Usage(GB)"] = (torch.cuda.memory_allocated()*world_size)/(1024**3)
    for i in range(num_iterations):
        print(f"run iteration: {i}")
        start_time = time.time()
        torch.cuda.synchronize()
        if collect_power_data:
            monitor.begin_window(window_key, sync_execution=True)
        pipelined_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
        if collect_power_data:
            mes = monitor.end_window("generate pipeline", sync_execution=True)
        torch.cuda.synchronize()
        end_time = time.time()

        generation_time = end_time - start_time
        tokens_generated = max_new_tokens*batch_size*num_micro_batches
        tps = (tokens_generated) / generation_time
        if collect_power_data: 
            timeline = power_monitor.get_power_timeline(
                power_domain="device_instant",  # or "device_average" or "memory_average"
                start_time=start_time,
                end_time=end_time)
            for gpu_idx, data in timeline.items():
                powers = [power_watts for timestamp, power_watts in data]
                if len(powers) != 0:
                    results['average_power_watts'].append(sum(powers) / len(powers))
                results['max_power_watts'].append(max(powers))
                results['min_power_watts'].append(min(powers))
            for gpu_idx in mes.gpu_energy: 
                results[f'gpu {gpu_idx} total_energy'].append(mes.gpu_energy[gpu_idx])

            results[f'total_energy_joules'].append(sum((mes.gpu_energy.values())))
            results[f'joules_per_token'].append(sum((mes.gpu_energy.values())) / tokens_generated)
        results['generation_time'].append(generation_time)
        results['tps'].append(tps)
    if collect_power_data:
        row['average_power_watts'] = statistics.mean(results['average_power_watts'])
        row['max_power_watts'] = statistics.mean(results['max_power_watts'])
        row['min_power_watts'] = statistics.mean(results['min_power_watts'])
        row['total_energy_joules'] = statistics.mean(results['total_energy_joules'])
        row['joules_per_token'] = statistics.mean(results['joules_per_token'])
        for gpu_idx in range(world_size):
            row[f'total energy from GPU: {gpu_idx}'] = statistics.mean(results[f'gpu {gpu_idx} total_energy'])

    row['tokens_per_second'] = statistics.mean(results['tps'])
    row['run_time_seconds'] = statistics.mean(results["generation_time"])
    

def time_to_first_token(pipelined_model, batch_size, seq_len, num_iterations, row, num_micro_batches, rank, use_dataset_prompts=False, model_id="ridger/MMfreeLM-2.7B"):
    """Estimate Time to First TOken """
    
    times_to_first_token = []
    micro_batches = []
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    generate_input_ids = generate_dataset_input_ids if use_dataset_prompts else generate_random_input_ids
    num_micro_batches = world_size
    if rank == 0: 
        for _ in range(num_micro_batches):
            inputs = generate_input_ids(model_id, batch_size, seq_len)
            micro_batches.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                })
    dist.barrier()

    pipelined_model.generate_pipelined(micro_batches[0:world_size], max_new_tokens=1)
    
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
        layers_multiplier, 
        vocab_multiplier, 
        collect_power_data=False,
        print_csv=False):

    first_row = True
    rank = int(os.environ.get("RANK", 0))
    from datetime import datetime
    current_time = datetime.now()
    if rank != 0:
        filename = f"whatever{rank}.csv"
    else:
        filename = f"outputs/csvs/pipelined_performance_eval-{current_time:%Y-%m-%d_%H:%M:%S}.csv"
    devices = ""
    for i in range(torch.cuda.device_count()): 
        devices += (torch.cuda.get_device_name(i))
        if i < torch.cuda.device_count()-1: devices += ","
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        original_hidden_layer_size = 2560
        original_num_layers = 32
        for weight_compression in [True]:
            pipelined_model = PipelineParallelMatMulFreeLM(
                weight_multiplier=weight_multiplier,
                layers_multiplier=layers_multiplier, 
                vocab_size_multiplier=vocab_multiplier, 
                weight_compression=weight_compression)  
            memory_usage = 0
            world_size = int(os.environ.get("WORLD_SIZE", 2))
            for device in range(world_size):
                print(f'Memory Allocated on Device: {device}: {torch.cuda.memory_allocated(device=device)}')
                memory_usage += torch.cuda.memory_allocated(device=device) 
            row = {
                'device': devices, 
                'Hidden Layer Size': original_hidden_layer_size*weight_multiplier, 
                'Number of Layers': original_num_layers*layers_multiplier, 
                'Weight Compression' : weight_compression}
            for batch_power in reversed(range(min_batch_power, max_batch_power)):
                batch_size = 2**batch_power
                row['Batch Size'] = batch_size
                print(f"\tCollecting data for batch size: {batch_size}")
                print(f"\t\tRunning Benchmarks...")
                benchmark_generation(pipelined_model, batch_size, sequence_length, iters, max_new_tokens, row, count_micro_batches, rank, collect_power_data=collect_power_data)
                time_to_first_token(pipelined_model, batch_size, sequence_length, iters, row, count_micro_batches, rank)
                if rank == 0:
                    if(first_row):
                        csvwriter = csv.DictWriter(csvfile, row.keys())
                        csvwriter.writeheader()
                        first_row = False
                    csvwriter.writerow(row) 
            del pipelined_model
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"Data written to {filename}")
    if print_csv:
        with open(filename, "r") as file:
            print(file.read())


def main():
    sequence_length=int(args.sequence_length)
    max_new_tokens=int(args.max_new_tokens)
    iters=int(args.iterations)
    min_batch_power=int(args.min_batch_power)
    max_batch_power=int(args.max_batch_power)
    count_micro_batches=int(args.micro_batches)
    weight_multiplier=float(args.weight_multiplier)
    layers_multiplier=float(args.layers_multiplier)
    vocab_multiplier=float(args.vocab_multiplier)
    collect_power_data= args.collect_power_data
    print_csv = args.print_csv

    create_csv_data(sequence_length, max_new_tokens, iters, min_batch_power, max_batch_power, count_micro_batches, weight_multiplier, layers_multiplier, vocab_multiplier, collect_power_data=collect_power_data, print_csv=print_csv)

if __name__ == "__main__":
    main()
