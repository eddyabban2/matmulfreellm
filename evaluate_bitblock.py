import torch
import torch.nn as nn
import copy 
import sys
import random
import gc 
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

from scaled_mmfree import base_config, ScalableHGRNBitModel


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
bitblock = HGRNBitBlock(config, 4)
init_weights(bitblock.attn.i_proj, config)
init_weights(bitblock.attn.f_proj, config)
init_weights(bitblock.attn.g_proj, config)
init_weights(bitblock.attn.o_proj, config)

init_weights(bitblock.mlp.gate_proj, config)
init_weights(bitblock.mlp.down_proj, config)

# hidden_states = torch.normal(mean=0, std=1.5, size=(10, 29, 500))
hidden_states = torch.rand(500,29, 2560, dtype=torch.float16)
bitblock.forward(hidden_states)





print("finished running ")
