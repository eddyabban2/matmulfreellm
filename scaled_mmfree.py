import torch
import torch.nn as nn
import copy 
import sys
import random
import gc 
from utils import generate_dataset_input_ids, create_string_from_tokens, generate_random_input_ids
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from mmfreelm.ops.fusedbitnet import CompressedType
import os
import psutil

def create_scaled_mmfree(
        layers_multiplier=1, 
        weight_multiplier=1, 
        vocab_size_multiplier=1, 
        weight_compression=False, 
        model_id="ridger/MMfreeLM-2.7B", 
        print_model_config=False, 
        device="cuda"
    ):
    if vocab_size_multiplier < 1:
        sys.exit("Vocab multiplier's smaller than 1 are unsupported")
    full_model = HGRNBitForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        ).cuda()
    if print_model_config:
        print_system_ram("System RAM After Loading init Model")
    full_model.to(device)
    if weight_multiplier != 1 or vocab_size_multiplier != 1: 
        hidden_size = int(2560*weight_multiplier)
        vocab_size = int(full_model.vocab_size*vocab_size_multiplier)
        del full_model.model.embeddings
        embeddings = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=hidden_size, 
            padding_idx=full_model.model.padding_idx, 
            device=device)
        nn.init.uniform_(embeddings.weight, a=-1, b=1)
        embeddings.to(torch.float16)
        full_model.model.embeddings = embeddings
    if print_model_config:
        print_system_ram("System RAM After updating embedding")
    weight_compression = CompressedType.NAIVE if weight_compression else CompressedType.FLOAT16
    if weight_multiplier != 1 or vocab_size_multiplier != 1:
        full_model.model.norm.increase_size(weight_multiplier)
        full_model.lm_head.increase_size(weight_multiplier, vocab_size_multiplier, compressed_type=weight_compression)
    if print_model_config:
        print_system_ram("System RAM After updating lm head and norm")
    layer_count = int(layers_multiplier * full_model.config.num_hidden_layers)
    new_hidden_size = int(2560*weight_multiplier)
    model_layers = []
    if layers_multiplier == 1:
        model_layers = [copy.deepcopy(full_model.model.layers[i])
            for i in range(full_model.config.num_hidden_layers)]
    else:
        for _ in range(layer_count):
            random_layer_index = random.randint(0, full_model.config.num_hidden_layers-1)
            model_layers.append(copy.deepcopy(full_model.model.layers[random_layer_index]))
    del full_model.model.layers
    print_system_ram(f"after deleting the original alyers")
    local_layers = nn.ModuleList(model_layers)
    local_layers = local_layers.to("cpu")
    full_model.model.lower_bounds = nn.Parameter(torch.rand(layer_count, new_hidden_size).to(device))

    if print_model_config:
        print(f"iterating over {len(local_layers)} layers")

    # if weight_multiplier != 1:
    for idx,layer in enumerate(local_layers):
        layer = layer.to(device)
        layer.attn.i_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.attn.f_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.attn.g_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.attn.o_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.mlp.gate_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.mlp.down_proj.increase_size(weight_multiplier, weight_multiplier, compressed_type=weight_compression)
        layer.attn_norm.increase_size(weight_multiplier)
        layer.mlp_norm.increase_size(weight_multiplier)
        layer.attn.g_norm.increase_size(weight_multiplier)
        if print_model_config:
            print_system_ram(f"increasing size {idx}")

    if print_model_config:
        print_system_ram("System RAM After loading layers")

    full_model.model.layers = local_layers
    torch.cuda.empty_cache()
    full_model.config.num_hidden_layers = len(local_layers)
    if vocab_size_multiplier != 1:
        full_model.config.vocab_size = int(full_model.config.vocab_size * vocab_size_multiplier)
    full_model.model.embeddings.to(device)
    full_model.model.layers.to(device)
    full_model.model.norm.to(device)
    full_model.lm_head.to(device)
    full_model = full_model.to(device)

    if hasattr(full_model, "generation_config"):
        full_model.generation_config.vocab_size = full_model.config.vocab_size
    if weight_multiplier != 1:
        full_model.config.hidden_size = new_hidden_size 
        
        if hasattr(full_model.config, "d_model"):
            full_model.config.d_model = new_hidden_size

    if print_model_config: 
        print(f"Embedding Layer: {full_model.model.embeddings}")
        print(f"Local Layers Count: {len(full_model.model.layers)} layers")
        print(f"Local Layers: {full_model.model.layers} layers")
        print(f"norm: {full_model.model.norm}")
        print(f"lm_head: {full_model.lm_head}")
    return full_model
process = psutil.Process(os.getpid())
def print_system_ram(label):
    mem_bytes = process.memory_info().rss 
    mem_gb = mem_bytes / (1024 ** 3)
    gpu_mem = torch.cuda.memory_allocated()
    gpu_gb = gpu_mem / (1024 **3)
    print(f"{label}: CPU {mem_gb:.2f} GB GPU: {gpu_gb:.2f} GB")

def main():

    MODEL_ID = "ridger/MMfreeLM-2.7B"
    # layers_multiplier = 1
    # weight_multiplier = 1
    # vocab_size_multiplier = 1
    layers_multiplier=2.5
    weight_multiplier=3.9375
    vocab_size_multiplier=4
    print_model_config = True
    use_weight_compression = True
    print_system_ram("System RAM Before Loading")
    model = create_scaled_mmfree(
        layers_multiplier=layers_multiplier, 
        weight_multiplier=weight_multiplier, 
        vocab_size_multiplier=vocab_size_multiplier, 
        model_id=MODEL_ID, 
        print_model_config=print_model_config, 
        weight_compression=use_weight_compression
    )
    print_system_ram("System RAM After Loading Model")

    base_memory_bytes = torch.cuda.memory_allocated()
    print(f"GPU Memory (Model Loading): {base_memory_bytes / (1024 ** 3):.2f} GB")

    batch_size = 5
    sequence_length = 20
    max_new_tokens = 8

    batch = generate_random_input_ids(MODEL_ID, batch_size, sequence_length)
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.4,
        temperature=0.6
    )

    print_system_ram("System RAM After Generation")

if __name__ == "__main__":
    main()
