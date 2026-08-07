import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_dataset_input_ids
import bitnet

model_id = "tiiuae/Falcon-E-1B-Base"

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16,
).to("cuda")

# Perform text generation
allocated = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"Allocated Memory Before Loading Models: {allocated:.2f} GB")

batch = generate_dataset_input_ids(model_id, 1, 100)
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()

_ = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.4,
    temperature=0.6)
