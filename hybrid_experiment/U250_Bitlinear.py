import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, logging
import torch
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, logging
import argparse
import nvtx
import transformers.integrations.bitnet as bitnet
import random
import numpy as np
import gc 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bitnet as local_bitnet
from utils import generate_random_input_ids, generate_dataset_input_ids, add_nvtx_hooks_to_every_module
from scaled_mmfree import print_system_ram
from mmfreelm.ops.fusedbitnet import CompressedType

from bitnet import BitLinear
bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear


class U250_BitLinear(BitLinear):
    def __init__(self, source_layer: BitLinear):
        print("Attempting to initialize U250 BitLinear layer")
        print(source_layer)


def main():
    print("eddy is here")
    bitnet_model_name = "microsoft/bitnet-b1.58-2B-4T"
    model = AutoModelForCausalLM.from_pretrained(bitnet_model_name).cuda()
    test_layer = U250_BitLinear(model.model.layers[0].self_attn.q_proj)


if __name__ == "__main__":
    main()
