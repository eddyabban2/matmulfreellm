import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('..')
import mmfreelm

# DeepSpeed requires local_rank environment variable or initializing via deepspeed runner
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2')) # e.g., 2 GPUs

model_id = "ridger/MMfreeLM-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model on meta or CPU first
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Initialize DeepSpeed Inference
model = deepspeed.init_inference(
    model,
    mp_size=world_size,             # Number of GPUs to split across
    dtype=torch.bfloat16,           # Match model weights
    replace_with_kernel_inject=False # MUST BE FALSE. MMfreeLM has custom Triton kernels; 
                                    # DeepSpeed's default Transformer injection will break it.
)

# DeepSpeed wraps the model, moving it to the correct local GPU device
input_prompt = "Eddy is"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(f"cuda:{local_rank}")

outputs = model.generate(input_ids, max_length=32, use_cache=False)
if local_rank == 0:
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))