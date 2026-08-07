import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
import mmfreelm
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_random_input_ids
from mmfreelm.ops.fusedbitnet import FusedBitLinear

def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factor
    scale = 1.0 / (w.abs().mean().clamp_(min=1e-5))
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1)
    return u
def prev_weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factor
    scale = 1.0 / (w.abs().mean().clamp_(min=1e-5))
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1)/scale
    return u

def main():
    
    model_name = 'ridger/MMfreeLM-2.7B'
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
    # print(model)
    for i in range(0,1):
        # print("printing model layers")
        tensor_list = [
            model.model.layers[i].attn.i_proj.weight,
            model.model.layers[i].attn.f_proj.weight,
            model.model.layers[i].attn.g_proj.weight,
            model.model.layers[i].mlp.gate_proj.weight,
            model.model.layers[i].mlp.down_proj.weight
        ]

        
        quant_atten_i = weight_quant(model.model.layers[i].attn.i_proj.weight)
        quant_atten_f = weight_quant(model.model.layers[i].attn.f_proj.weight)
        quant_atten_g = weight_quant(model.model.layers[i].attn.g_proj.weight)
        quant_mlp_gate = weight_quant(model.model.layers[i].mlp.gate_proj.weight)
        quant_mlp_down = weight_quant(model.model.layers[i].mlp.down_proj.weight) 
        quant_tensor = [
            quant_atten_i, 
            quant_atten_f, 
            quant_atten_g, 
            quant_mlp_gate,
            quant_mlp_down
        ]

        prev_quant_atten_i = prev_weight_quant(model.model.layers[i].attn.i_proj.weight)
        prev_quant_atten_f = prev_weight_quant(model.model.layers[i].attn.f_proj.weight)
        prev_quant_atten_g = prev_weight_quant(model.model.layers[i].attn.g_proj.weight)
        prev_quant_mlp_gate = prev_weight_quant(model.model.layers[i].mlp.gate_proj.weight)
        prev_quant_mlp_down = prev_weight_quant(model.model.layers[i].mlp.down_proj.weight) 
        prev_quant_tensor = [
            prev_quant_atten_i, 
            prev_quant_atten_f, 
            prev_quant_atten_g, 
            prev_quant_mlp_gate,
            prev_quant_mlp_down
        ]

        all_weights_flat = torch.cat([t.flatten() for t in tensor_list])
        quant_flat_tensor = torch.cat([t.flatten() for t in quant_tensor])
        prev_quant_flat_tensor = torch.cat([t.flatten() for t in prev_quant_tensor])
        print(f"Total elements combined: {all_weights_flat.numel()}")
        weights_np = all_weights_flat.cpu().detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(weights_np, bins=50, density=True, alpha=0.7, label="Combined Weights")
        plt.title("Histogram of Unquantized Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Probability Density")
        plt.savefig(f'Layer {i} weight distribution(Unquantized).png')

        weights_np = quant_flat_tensor.cpu().detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(weights_np, bins=50, density=True, alpha=0.7, label="Combined Weights")
        plt.title("Histogram of Weights Quantized without Scale Division")
        plt.xlabel("Weight Value")
        plt.ylabel("Probability Density")
        plt.savefig(f'Layer {i} weight distribution(Quantized without Scale Division).png')

        weights_np = prev_quant_flat_tensor.cpu().detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(weights_np, bins=50, density=True, alpha=0.7, label="Combined Weights")
        plt.title("Histogram of Weights Quantized with Scale Division")
        plt.xlabel("Weight Value")
        plt.ylabel("Probability Density")
        plt.savefig(f'Layer {i} weight distribution(Quantized with Scale Division).png')

if __name__ == "__main__":
    main()