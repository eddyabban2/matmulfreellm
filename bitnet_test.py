from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import generate_dataset_input_ids
import transformers.integrations.bitnet as bitnet
import inspect
import bitnet as eddy_bitnet


original_bitnet_function = bitnet.pack_weights
bitnet.pack_weights = eddy_bitnet.pack_weights

original_bitnet_function = bitnet.unpack_weights
bitnet.unpack_weights = eddy_bitnet.unpack_weights

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
 device_map="auto")

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
allocated = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"Allocated Memory Before Loading Models: {allocated:.2f} GB")

batch = generate_dataset_input_ids(model_id, 10, 100)
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()
print("model")

_ = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.4,
    temperature=0.6)
