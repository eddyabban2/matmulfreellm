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
    default=1,
    help="sets the number of new tokens to be generated"
)

parser.add_argument(
    "-i",
    "--iterations",
    default=1,
    help="Determines the number of iterations to benchmark for"
)

args = parser.parse_args()
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

num_iterations = int(args.iterations)
batch_size = int(args.batch_size)
seq_len = int(args.seq_len)
max_new_tokens = int(args.max_new_tokens)
MODEL_NAME = 'ridger/MMfreeLM-2.7B'
batch = None
rank_string = f"[{int(os.environ.get("RANK", 0))}]: "
print(rank_string + "pipelined quiet run is running")

pipeline_model = PipelineParallelMatMulFreeLM(MODEL_NAME)  

if int(os.environ.get("RANK", 0)) == 0:
    if args.use_dataset_prompts:
        batch = generate_dataset_input_ids(MODEL_NAME, batch_size, seq_len)
    else: 
        batch = generate_random_input_ids(MODEL_NAME, batch_size, seq_len)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
else: 
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long) 
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
dist.barrier()



print(rank_string + "warmup running")
with nvtx.annotate("warmup", color="white"):
    pipeline_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
print(rank_string + "warmup finished")
with nvtx.annotate("workload", color="cyan"):
    for _ in range(num_iterations):
        pipeline_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        dist.barrier()
    dist.barrier()
dist.barrier()
print(rank_string + "inference worked")
dist.destroy_process_group()

