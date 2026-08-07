from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
from utils import generate_dataset_input_ids
import transformers.integrations.bitnet as bitnet
import bitnet as local_bitnet
import gc 

logging.set_verbosity_error()
logging.disable_default_handler()
logging.disable_propagation()

bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear

model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id).cuda()
print(f"Bitnet: {model}")
# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
allocated = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"Allocated Memory After Loading Models: {allocated:.2f} GB")

batch = generate_dataset_input_ids(model_id, 10, 100)
print(f"Allocated Memory After Loading input prompts: {allocated:.2f} GB")
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()
print("model")

def estimate_model_memory(model, verbose=True):
    total_bytes = 0
    layer_summary = {}

    total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    for name, param in model.named_parameters():
        bytes_used = param.numel() * param.element_size()  # element_size() = bytes per dtype
        total_bytes += bytes_used
        
        # Group by top-level module
        top = name.split(".")[0]
        layer_summary[top] = layer_summary.get(top, 0) + bytes_used

        if verbose:
            print(f"{name:60s} | {str(param.shape):30s} | {param.dtype} | {bytes_used/1e6:.2f} MB")

    print("\n--- Summary by top-level module ---")
    for k, v in layer_summary.items():
        print(f"  {k:20s}: {v/1e6:.2f} MB")
    
    print(f"\nTotal  memory : {total_bytes/1e9:.3f} GB")
    print(f"\nTotal parameter memory : {total_param_bytes/1e9:.3f} GB")
    print(f"\nTotal buffer memory : {total_buffer_bytes/1e9:.3f} GB")
    return total_bytes
gc.collect()
torch.cuda.empty_cache()
# Also check live GPU allocation (if on CUDA)
print(f"GPU allocated before warmup: {torch.cuda.memory_allocated()/1e9:.3f} GB")
print(f"GPU reserved  before warmup: {torch.cuda.memory_reserved()/1e9:.3f} GB")

# HuggingFace convenience method (if using transformers)
# print(f"model size memory: {estimate_model_memory(model,verbose=True)}")


_ = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1,
    do_sample=True,
    top_p=0.4,
    temperature=0.6)
gc.collect()
torch.cuda.empty_cache()
print(f"GPU allocated after warmup: {torch.cuda.memory_allocated()/1e9:.3f} GB")
print(f"GPU reserved  after warmup: {torch.cuda.memory_reserved()/1e9:.3f} GB")

for _ in range(10):
    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.4,
        temperature=0.6)
