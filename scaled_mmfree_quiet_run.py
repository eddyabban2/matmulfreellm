# Example run: 
# python scaled_mmfree_quiet_run.py -b 5 -s 10 -n 10 -i 1

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, logging
from utils import generate_random_input_ids, generate_dataset_input_ids, add_nvtx_hooks_to_every_module
import argparse
import nvtx
import transformers.integrations.bitnet as bitnet
import bitnet as local_bitnet
import random
import numpy as np
import gc 
from scaled_mmfree import create_scaled_mmfree, print_system_ram

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# Force cuDNN determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

parser.add_argument(
    "--prefill_decode",
    action='store_true',
    default=False,
    help="sets whether we mark prefill and decode sections of the workload"
)

print("quiet run is running")
args = parser.parse_args()

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

model_name = "ridger/MMfreeLM-2.7B"
num_iterations = int(args.iterations)
batch_size = int(args.batch_size)
seq_len = int(args.seq_len)
max_new_tokens = int(args.max_new_tokens)
prefill_decode = args.prefill_decode

batch = None

if args.use_dataset_prompts:
    batch = generate_dataset_input_ids(model_name, batch_size, seq_len)
else: 
    batch = generate_random_input_ids(model_name, batch_size, seq_len)
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()

layers_multiplier=2.5
weight_multiplier=3.9375
vocab_size_multiplier=4
weight_compression=True
model_id="ridger/MMfreeLM-2.7B"
print_model_config=True
device="cuda"
print("attempting to load model")
model = create_scaled_mmfree(
    layers_multiplier=layers_multiplier,
    weight_multiplier=weight_multiplier, 
    vocab_size_multiplier=vocab_size_multiplier, 
    weight_compression=weight_compression, 
    model_id=model_id, 
    print_model_config=print_model_config, 
    device=device)
print("finished loading model")
add_nvtx_hooks_to_every_module(model)
print("warmup running")
with nvtx.annotate("warmup", color="white"):
    # run a warm up generate
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1)
torch.cuda.synchronize()
gc.collect()
torch.cuda.synchronize()
torch.use_deterministic_algorithms(True, warn_only=True)
print("warmup finished")
print_system_ram("Memory After Warmup")
#generate call
with nvtx.annotate("workload", color="cyan"):
    if prefill_decode:
        with nvtx.annotate("prefill", color="red"):
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        with nvtx.annotate("switching between pre and deco", color="green"):
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(-1)
            print("switching now")
        with nvtx.annotate("decode", color="blue"):
            with torch.no_grad():
                for i in range(max_new_tokens-1):
                    with nvtx.annotate(f"decodingStep{i}", color="cyan"):
                        out = model(input_ids=next_tok, past_key_values=past,
                                    use_cache=True, return_dict=True)
                        past = out.past_key_values
                        next_tok = out.logits[:, -1:, :].argmax(-1)
    else: 
        for _ in range(num_iterations):
            model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_new_tokens=max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
print("inference worked")
