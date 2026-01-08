import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from bench_utils import generate_random_input_ids
import transformers
import argparse
import statistics
from zeus.monitor import ZeusMonitor, PowerMonitor
import csv

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
    "-s", 
    "--seq_len",
    default=1,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
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

parser.add_argument(
    "--model_name", 
    default='ridger/MMfreeLM-2.7B', 
    help="sets the model name to be used"
)

args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

# 2. Suppress HuggingFace Hub logging (where shard-loading prints originate)
logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

model_name = args.model_name
num_iterations = int(args.iterations)
batch_size = int(args.batch_size)
seq_len = int(args.seq_len)
max_new_tokens = int(args.max_new_tokens)

batch = generate_random_input_ids(model_name, batch_size, seq_len)
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
            


# run a warm up generate 
_ = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    top_p=0.4,
    temperature=0.6)

#generate call
for _ in range(num_iterations):
    _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.4,
            temperature=0.6
        )
