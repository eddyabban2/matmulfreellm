import torch
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
sys.path.append('..')
from transformers import AutoModelForCausalLM, AutoTokenizer
import mmfreelm

model_id = "ridger/MMfreeLM-2.7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# "auto" will automatically split the layers across all available GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto"
)
print(model)
# tokenizer.pad_token = tokenizer.eos_token
# # print(model)

# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '2')) # e.g., 2 GPUs

# device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
# torch.distributed.init_process_group(rank=local_rank, world_size=world_size)
# input_prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
# input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))