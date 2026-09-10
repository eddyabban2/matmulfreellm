import torch
import torch.nn as nn
import copy 
import sys
import random
import gc 
from typing import List, Optional, Tuple, Union
import math
from utils import generate_dataset_input_ids, create_string_from_tokens, generate_random_input_ids
from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig
from mmfreelm.ops.fusedbitnet import CompressedType
from transformers import AutoModelForCausalLM
import os
import psutil
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitModel, HGRNBitPreTrainedModel, HGRNBitBlock
from mmfreelm.ops.fusedbitnet import FusedBitLinear
from mmfreelm.modules import RMSNorm
import nvtx
import statistics

from scaled_mmfree import base_config, ScalableHGRNBitModel

import torch
import nvtx
from typing import Optional, Tuple, List, Dict

class EvluationBitBlock(HGRNBitBlock):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            lower_bound: Optional[torch.Tensor] = False,
            **kwargs,
        ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, float]]:
        
        # Helper to create timing events
        def create_event():
            return torch.cuda.Event(enable_timing=True)
            
        # Dictionary to store start and end events for each operation
        events = {
            "attn_norm": (create_event(), create_event()),
            "attn": (create_event(), create_event()),
            "mlp_norm": (create_event(), create_event()),
            "mlp": (create_event(), create_event()),
            "residual_add": (create_event(), create_event()),
        }

        with nvtx.annotate("HGRNBitBlock forward", color="cornsilk"):
            residual = hidden_states
            
            # 1. Benchmark attn_norm
            events["attn_norm"][0].record()
            hidden_states = self.attn_norm(hidden_states)
            events["attn_norm"][1].record()
            
            # 2. Benchmark attn
            events["attn"][0].record()
            hidden_states, attentions, past_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lower_bound=lower_bound
            )
            events["attn"][1].record()
            
            events["mlp_norm"][0].record()
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
            events["mlp_norm"][1].record()
            
            events["mlp"][0].record()
            hidden_states = self.mlp(hidden_states)
            events["mlp"][1].record()
            
            events["residual_add"][0].record()
            hidden_states = residual + hidden_states
            events["residual_add"][1].record()

            torch.cuda.synchronize()
            
            timings = {
                op: start.elapsed_time(end) 
                for op, (start, end) in events.items()
            }

            outputs = (hidden_states, attentions, past_key_values)
            
            return (outputs, timings)

def init_weights(
        module: nn.Module,
        config,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, FusedBitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * config.num_hidden_layers)
torch.set_default_dtype(torch.float16)
config = base_config
layer_idx = 5
bitblock = EvluationBitBlock(config, layer_idx).cuda()
init_weights(bitblock.attn.i_proj, config)
init_weights(bitblock.attn.f_proj, config)
init_weights(bitblock.attn.g_proj, config)
init_weights(bitblock.attn.o_proj, config)

init_weights(bitblock.mlp.gate_proj, config)
init_weights(bitblock.mlp.down_proj, config)

# hidden_states = torch.normal(mean=0, std=1.5, size=(10, 29, 500))
hidden_states = torch.rand(500,29, 2560, dtype=torch.float16).to("cuda")
print("Running warmup")
with torch.no_grad():
    bitblock.forward(hidden_states)
print("Collecting results")

loops = 10000
all_results = {}
for _ in range(loops):
    with torch.no_grad():
        results = bitblock.forward(hidden_states)
    for key, value in results[1].items():
        if key in all_results: 
            all_results[key].append(value)
        else: 
            all_results[key] = [value]

import matplotlib.pyplot as plt

# Create a figure with a subplot for each operation
fig, axes = plt.subplots(nrows=len(all_results), ncols=1, figsize=(10, 2.5 * len(all_results)))
fig.tight_layout(pad=4.0)

for ax, (key, values) in zip(axes, all_results.items()):
    # Plot histogram with 20 bins
    ax.hist(values, bins=20, alpha=0.7, color='royalblue', edgecolor='black')
    ax.set_title(f"{key} Execution Time", fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency")
    
    # Add a vertical line for the mean
    mean_val = statistics.mean(values)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    ax.text(mean_val, ax.get_ylim()[1]*0.9, f' Mean: {mean_val:.3f}ms', color='red')

plt.suptitle("EvluationBitBlock Timing Distributions", fontsize=14, y=1.02)

# Save the plot (crucial if running on a headless GPU server)
plt.savefig("outputs/images/timgs.png", bbox_inches='tight')

# print(f"full results: {all_results}")
for key, values in all_results.items(): 
    print("=========================================")
    print(f"{key}")
    print("-----------------------------------------")
    print(f"Mean: {statistics.mean(values)}")
    print(f"Standard Deviation: {statistics.stdev(values)}")
    print(f"Min: {min(values)}")
    print(f"Max: {max(values)}")

print("finished running")
