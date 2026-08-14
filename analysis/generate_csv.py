"""
Creates a CSV file with benchmark results for MMFreeLM models.

Example usage:
    python generate_csv.py -s 32 --max_new_tokens 32 -i 15 --min_batch_power 0 --max_batch_power 12
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import time
import torch
import gc
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from mmfreelm.benchmark.utils import generate_random_input_ids, generate_dataset_input_ids
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv
import transformers.integrations.bitnet as bitnet
from mmfreelm.integrations import bitnet as local_bitnet
import mmfreelm
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from mmfreelm.ops.fusedbitnet import CompressedType

bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear

parser = argparse.ArgumentParser(
    description="creates a csv file with benchmark results"
)

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

def profile_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(num_iterations):
            _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
                )
    return prof

def benchmark_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):
    """Run benchmark with multiple prompts and iterations."""
    
    results = {
            'tps': [],
            'generation_time': [],
            'average_power_watts': [],
            'max_power_watts': [],
            'min_power_watts': [],
            'total_energy_joules': [],
            'energy_per_iteration_joules': [],
            'joules_per_token': []
        }
    batch = None
    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
        batch = generate_random_input_ids(model_name, batch_size, seq_len)

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # power_monitor = PowerMonitor(gpu_indices=[torch.cuda.current_device()])
    # monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    # window_key = f"Batch Size {batch_size} Seq Len {seq_len}"

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    for iter in range(num_iterations):
        start_time = time.time()
        torch.cuda.synchronize()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
        torch.cuda.synchronize()
        end_time = time.time()

        generation_time = end_time - start_time
        tokens_generated = (outputs.shape[1] - seq_len)*batch_size
        tps = (tokens_generated) / generation_time

        results['generation_time'].append(generation_time)
        results['tps'].append(tps)
        
    row['tokens_per_second'] = statistics.mean(results['tps'])
    row['run_time_seconds'] = statistics.mean(results["generation_time"])

def detailed_runtime_metrics(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):

    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
        batch = generate_random_input_ids(model_name, batch_size, seq_len)

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    prefill_times = []
    decode_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        end_time = time.time()

        prefill_time = end_time - start_time
        prefill_times.append(prefill_time)
        curr_decode_times = []
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)
        with torch.no_grad():
            for i in range(max_new_tokens-1):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    out = model(input_ids=next_tok, past_key_values=past,
                                use_cache=True, return_dict=True)
                    past = out.past_key_values
                    next_tok = out.logits[:, -1:, :].argmax(-1)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    curr_decode_times.append(end_time - start_time)
            decode_times.append(curr_decode_times)
    row["Avg Prefill Time (s)"] = statistics.mean(prefill_times)
    row["Max Prefill Time (s)"] = max(prefill_times)
    row["Min Prefill Time (s)"] = min(prefill_times)
    all_single_decode_times = [t for iteration in decode_times for t in iteration]
    total_decode_times = [sum(iteration) for iteration in decode_times]
    row["Avg Single Deocde Time (s)"] = statistics.mean(all_single_decode_times)
    row["Min Single Deocde Time (s)"] = min(all_single_decode_times)
    row["Max Single Deocde Time (s)"] = max(all_single_decode_times)
    row[f"Avg Deocde Time For {max_new_tokens-1} Tokens (s)"] = statistics.mean(total_decode_times)
    row[f"Max Deocde Time For {max_new_tokens-1} Tokens (s)"] = max(total_decode_times)
    row[f"Min Deocde Time For {max_new_tokens-1} Tokens (s)"] = min(total_decode_times)

def first_token_time(model, batch_size, seq_len, num_iterations, model_name='ridger/MMfreeLM-2.7B', use_dataset_prompts=False):
    """Run benchmark with multiple prompts and iterations."""
    if use_dataset_prompts:
        batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
    else:    
        batch = generate_random_input_ids(model_name, batch_size, seq_len)

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    times = []
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=seq_len+5,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    

    for iter in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=1,
            max_new_tokens=1,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
        torch.cuda.synchronize()
        end_time = time.time()
        generation_time = end_time - start_time
        times.append(generation_time)
    return times

def profile_generation(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B'):
    # create random input tokens
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
    
    # profile generate 
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(num_iterations):
            _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.4,
                    temperature=0.6
                )
    events = prof.key_averages()
    table_string = prof.key_averages().table().split('\n')
    cpu_time = table_string[-3].split()[4]
    if cpu_time.endswith('ms'):
        cpu_time = float(cpu_time[:-2]) / 1e3
    else:
        cpu_time = float(cpu_time[:-1])
    cuda_time = table_string[-2].split()[4]
    if cuda_time.endswith('ms'):
        cuda_time = float(cuda_time[:-2]) / 1e3
    else:
        cuda_time = float(cuda_time[:-1])
    flops = sum(e.flops for e in events) / float(row['run_time_seconds'])
    row["FLOPS"] = flops
    row ["CPU_time_seconds"] = cpu_time
    row["CUDA_time_seconds"] = cuda_time

def run_warmup(model, model_name):
    batch = generate_random_input_ids(model_name, 1, 1)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # run a warm up generate 
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)

def _pynvml_power_samples(generate_fn, sample_interval_s=0.05):
    """Fallback instantaneous power sampling via NVML while generate_fn runs."""
    import threading
    try:
        import pynvml
    except ImportError:
        from nvidia_ml_py import pynvml  # type: ignore

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(torch.cuda.current_device()))
    samples = []
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            try:
                samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(sample_interval_s)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    start = time.time()
    try:
        generate_fn()
    finally:
        stop.set()
        thread.join(timeout=2.0)
        torch.cuda.synchronize()
        end = time.time()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return samples, end - start


def get_power_data(model, batch_size, seq_len, num_iterations, max_new_tokens, row, model_name='ridger/MMfreeLM-2.7B'):
    """Fill Zeus (or NVML-fallback) power/energy columns on ``row``."""
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    def _run_iters():
        for _ in range(num_iterations):
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.4,
                temperature=0.6,
            )

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_new_tokens=1,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6,
    )

    tokens = batch_size * max_new_tokens * num_iterations
    powers = []
    energy_j = None
    source = "zeus"

    try:
        power_monitor = PowerMonitor(gpu_indices=[torch.cuda.current_device()])
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
        window_key = f"Batch Size {batch_size} Seq Len {seq_len}"
        start_time = time.time()
        monitor.begin_window(window_key, sync_execution=True)
        _run_iters()
        mes = monitor.end_window(window_key, sync_execution=True)
        end_time = time.time()
        timeline = power_monitor.get_power_timeline(
            power_domain="device_instant",
            gpu_index=0,
            start_time=start_time,
            end_time=end_time,
        )
        for _gpu_idx, data in timeline.items():
            powers.extend(power_watts for _ts, power_watts in data)
        energy_j = float(mes.gpu_energy[0])
    except Exception as err:
        print(f"\t\tZeus power capture failed ({err}); falling back to NVML sampling.")
        source = "pynvml"
        powers, elapsed = _pynvml_power_samples(_run_iters)
        if powers:
            energy_j = (sum(powers) / len(powers)) * elapsed

    if powers:
        row["average_power_watts"] = sum(powers) / len(powers)
        row["max_power_watts"] = max(powers)
        row["min_power_watts"] = min(powers)
    else:
        row["average_power_watts"] = None
        row["max_power_watts"] = None
        row["min_power_watts"] = None
        print("\t\tWARNING: no power samples collected")

    if energy_j is not None:
        row["total_energy_joules"] = energy_j
        row["energy_per_iteration_joules"] = energy_j / max(num_iterations, 1)
        row["joules_per_token"] = energy_j / max(tokens, 1)
    else:
        row["total_energy_joules"] = None
        row["energy_per_iteration_joules"] = None
        row["joules_per_token"] = None

    row["power_source"] = source
    print(
        f"\t\tPower ({source}): avg={row['average_power_watts']} W "
        f"energy={row['total_energy_joules']} J"
    )

def set_ridger_compression(compression, model):
    for layer in model.model.layers: 
        layer.set_compression(compression)
def set_bitnet_compression(compression, model):
    compression = (compression == CompressedType.NAIVE) 
    for layer in model.model.layers:
        layer.self_attn.q_proj.compress_weights = compression
        layer.self_attn.k_proj.compress_weights = compression
        layer.self_attn.v_proj.compress_weights = compression
        layer.self_attn.o_proj.compress_weights = compression

        layer.mlp.gate_proj.compress_weights = compression
        layer.mlp.up_proj.compress_weights = compression
        layer.mlp.down_proj.compress_weights = compression

def get_batch_sizes():
    if args.batch_sampling == "powers-of-two":
        return [2 ** power for power in range(int(args.min_batch_power), int(args.max_batch_power))]

    min_size = int(args.min_batch_size)
    max_size = int(args.max_batch_size)
    samples = int(args.batch_samples)
    if min_size < 1 or max_size < min_size:
        raise ValueError("batch sizes must satisfy 1 <= min_batch_size <= max_batch_size")
    if samples < 1:
        raise ValueError("batch_samples must be at least 1")
    if samples == 1 or min_size == max_size:
        return [min_size]

    log_min = math.log(min_size)
    log_max = math.log(max_size)
    sizes = {
        int(round(math.exp(log_min + (log_max - log_min) * index / (samples - 1))))
        for index in range(samples)
    }
    return sorted(sizes | {min_size, max_size})


def is_cuda_oom(error):
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or any(
        marker in message
        for marker in ("out of memory", "nvmapmemalloc", "nvml_success", "cudacachingallocator")
    )


def csv_output_path(prefix: str) -> str:
    from datetime import datetime
    out_dir = os.environ.get("OUTPUT_DIR", "outputs/csvs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{prefix}-{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")


def create_csv_data(sequence_length, iters, max_new_tokens, model_name='ridger/MMfreeLM-2.7B'):
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Collecting Data to be used in a CSV")
    print(f"Branch/tag hint: {os.environ.get('GIT_BRANCH', 'local')}")
    first_row = True
    filename = csv_output_path("benchmark_results")
    with open(filename, 'w') as csvfile:
        csvwriter = None  
        row = {
            'device': device,
            'model': model_name,
            'git_branch': os.environ.get('GIT_BRANCH', 'local'),
            'gpu_resource': os.environ.get('GPU_RESOURCE', ''),
        }
        print(f"Collecting data for model: {model_name}")
        if model_name == "ridger/MMfreeLM-2.7B":
            from mmfreelm.packed_loader import load_packed_hgrn

            compression_modes = [True]
        else:
            compression_modes = [CompressedType.FLOAT16, CompressedType.NAIVE]

        for packed in compression_modes:
            if model_name == "ridger/MMfreeLM-2.7B":
                model = load_packed_hgrn(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).cuda()
            if model_name != "ridger/MMfreeLM-2.7B" and "ridger" in model_name:
                model = model.half()
                set_ridger_compression(packed, model)
            if "bitnet" in model_name:
                set_bitnet_compression(packed, model)
            row["Weight Packing"] = packed
            run_warmup(model, model_name)
            gc.collect()
            torch.cuda.empty_cache()
            row["Memory Usage"] = torch.cuda.memory_allocated() / (1024**3)
            batch_sizes = get_batch_sizes()
            print(f"\tBatch sizes: {batch_sizes}")
            for batch_size in batch_sizes:
                row['batch size'] = batch_size
                print(f"\tCollecting data for batch size: {batch_size}")
                try:
                    print(f"\t\tRunning Benchmarks...")
                    start_time = time.time()
                    benchmark_generation(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=False)
                    end_time = time.time()
                    print(f"\t\t\tBenchmarks completed in {end_time-start_time} sec")

                    start_time = time.time()
                    detailed_runtime_metrics(model, batch_size, sequence_length, iters, max_new_tokens, row, model_name=model_name, use_dataset_prompts=False)
                    end_time = time.time()
                    print(f"\t\tPrefill and Decode Times completed in {end_time-start_time} sec")
                    if args.collect_power_data:
                        print(f"\t\tCollecting power metrics...")
                        start_time = time.time()
                        get_power_data(
                            model,
                            batch_size,
                            sequence_length,
                            iters,
                            max_new_tokens,
                            row,
                            model_name=model_name,
                        )
                        print(f"\t\t\tPower metrics completed in {time.time()-start_time} sec")
                except RuntimeError as error:
                    if not is_cuda_oom(error):
                        raise
                    print(f"\tCUDA out of memory at batch size {batch_size}; ending this sweep.")
                    torch.cuda.empty_cache()
                    break
                if(first_row):
                    csvwriter = csv.DictWriter(csvfile, row.keys())
                    csvwriter.writeheader()
                    first_row = False
                csvwriter.writerow(row)
                csvfile.flush()
                os.fsync(csvfile.fileno()) 
            del model
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Data written to {filename}")
    if args.print_csv:
        with open(filename, "r") as file:
            print(file.read())

def main():
    if args.fixed_point:
        print("fixed point not yet supported")
        quit()

    sequence_length=int(args.sequence_length)
    iters=int(args.iterations)
    max_new_tokens=int(args.max_new_tokens)
    
    create_csv_data(sequence_length, iters, max_new_tokens, model_name=args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="creates a csv file with benchmark results"
    )

    parser.add_argument(
        "-s", 
        "--sequence_length",
        default=32,
        help="sets the sequence length of input tokens"
    )

    parser.add_argument( 
        "--max_new_tokens",
        default=2,
        help="sets the sequence length of input tokens"
    )

    parser.add_argument(
        "-i", 
        "--iterations",
        default=5,
        help="Determines the number of iterations to benchmark for"
    )

    parser.add_argument (
        "-f",
        "--fixed_point",
        action="store_true",
        help="Switches the model to fixed point",
    )

    parser.add_argument(
        "--min_batch_power", 
        default=0,
        help="stores the minimum batch power to go up to when profiling",
    )

    parser.add_argument(
        "--max_batch_power", 
        default=1,
        help="stores the maximum batch power to go up to when profiling",
    )

    parser.add_argument(
        "--batch_sampling",
        choices=("exponential", "powers-of-two"),
        default="powers-of-two",
        help="batch-size sampling strategy (default: powers of two from min/max_batch_power)",
    )

    parser.add_argument(
        "--min_batch_size",
        default=1,
        help="smallest batch size for exponential sampling",
    )

    parser.add_argument(
        "--max_batch_size",
        default=8192,
        help="largest batch size to attempt for exponential sampling",
    )

    parser.add_argument(
        "--batch_samples",
        default=25,
        help="number of exponentially spaced batch sizes to attempt",
    )

    parser.add_argument(
        "--use_original",
        action='store_true',
        default=False,
        help="changes the model to using the original implementation"
    )

    parser.add_argument(
        "--print_csv",
        action='store_true',
        default=False,
        help="prints csv after creating data"
    )

    parser.add_argument(
        "--model", 
        default="ridger/MMfreeLM-2.7B",
        help="selects model",
    )

    parser.add_argument(
        "--collect_power_data",
        action='store_true',
        default=False,
        help="changes whether we collect power data"
    )

    args = parser.parse_args()
    main()
