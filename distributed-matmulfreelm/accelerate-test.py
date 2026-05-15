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
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

input_prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))