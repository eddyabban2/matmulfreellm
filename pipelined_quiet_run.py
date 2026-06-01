# Example Usage: If you have 2 GPUs
#   torchrun --nproc_per_node=2 pipelined_quiet_run.py 

import os
import torch
import torch.distributed as dist
import argparse
import nvtx
from transformers import AutoModelForCausalLM, logging
from utils import generate_random_input_ids, generate_dataset_input_ids
from pipeline_mmfreelm import PipelineParallelMatMulFreeLM

parser = argparse.ArgumentParser(
    description="performs Batched Generation"
)

parser.add_argument(
    "-b",
    "--batch_size",
    default=1,
    help="sets the batch size"
)

parser.add_argument(
    "--use_dataset_prompts",
    action='store_true',
    default=False,
    help="changes whether we are using random prompts or dataset prompts"
)

parser.add_argument(
    "-s",
    "--seq_len",
    default=1,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-n",
    "--max_new_tokens",
    default=3,
    help="sets the number of new tokens to be generated"
)

parser.add_argument(
    "-m",
    "--num_micro_batches",
    default=5,
    help="sets the number of micro batches"
)

parser.add_argument(
    "-i",
    "--iterations",
    default=1,
    help="Determines the number of iterations to benchmark for"
)

parser.add_argument(
    "-l", 
    "--layers_multiplier",
    default=1,
    help="set the amount of times to multiply the number of layers"
)
parser.add_argument(
    "-w", 
    "--weight_multiplier",
    default=1,
    help="set the amount of times to multiply the number of weights in each layer"
)

args = parser.parse_args()
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

num_iterations = int(args.iterations)
batch_size = int(args.batch_size)
seq_len = int(args.seq_len)
max_new_tokens = int(args.max_new_tokens)
num_micro_batches = int(args.num_micro_batches)
weight_multiplier = int(args.weight_multiplier)
layers_multiplier = int(args.layers_multiplier)
MODEL_NAME = 'ridger/MMfreeLM-2.7B'
batch = None
rank_string = f"[{int(os.environ.get("RANK", 0))}]: "
print(rank_string + "pipelined quiet run is running")
pipeline_model = PipelineParallelMatMulFreeLM(layers_multiplier=layers_multiplier, weight_multiplier=weight_multiplier)  
micro_batches = []
if int(os.environ.get("RANK", 0)) == 0:
    for _ in range(num_micro_batches):
        inputs = generate_dataset_input_ids(MODEL_NAME, batch_size, seq_len)
        micro_batches.append(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )
dist.barrier()

print(rank_string + "warmup running")
with nvtx.annotate("warmup", color="white"):
    pipeline_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
print(rank_string + "warmup finished")
with nvtx.annotate("workload", color="cyan"):
    for _ in range(num_iterations):
        pipeline_model.generate_pipelined(micro_batches, max_new_tokens=max_new_tokens)
        dist.barrier()
    dist.barrier()
dist.barrier()
print(rank_string + "inference worked")
dist.destroy_process_group()

