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


import time
import numpy as np
import pynq

from ternary_matmul.sw_utils.lib.asm import Asm
from ternary_matmul.sw_utils.lib.config import Config
from ternary_matmul.sw_utils.lib.pynqvivado_ternip import PynqvivadoTernip
from ternary_matmul.sw_utils.lib.pynqvivado_common import (
    build_pynq_overlay,
    configs_from_target,
    enumerate_cu_names,
)
from ternary_matmul.sw_utils.lib.huggingface import HuggingFace
from ternary_matmul.sw_utils.lib.algorithm_tree import AlgorithmTree
from ternary_matmul.sw_utils.lib.matmulfree_algorithm_tree import (
    matmulfree_algorithm_tree,
    matmulfree_zero_initialized_abstract_memory,
)

from bitnet import BitLinear
bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear


    


class U250_BitLinear(BitLinear):
    def bitlinear_algorithm_tree():
        print("attempting to build bit linear algrothim tree")
    def create_algorithm_tree(config):
        def load_vector(size, address_label):
                vec = tree.new_abstract_vector(size)
                tree.new_abstract_operation('ldv', [], [vec], {'address_label': address_label})
                return vec

        tree = AlgorithmTree(config)
        input_vector = load_vector(config.input_features)
        return tree
    def compile_kernel_stream(self, config_path, batch_size, source_layer):
        config = Config(config_path)
        config.BatchSize = batch_size
        config.input_dim = source_layer.input_dim
        config.output_dim = source_layer.output_dim

        tree = self.create_algorithm_tree(config)
        # we do not using a hugging face variable for interacting with hugging face here
        return "eddy was here"
    def __init__(self, source_layer: BitLinear, unique_specs):
        results = [self.compile_kernel_stream(config_path, batch_size, source_layer) for (config_path, batch_size) in unique_specs]
        print(f"got the results: {results}")

def main():
    args = init_parser()

    compute_unit_list = configs_from_target(args.config)
    configs = [config for (config, _bank, _slr, _name) in compute_unit_list]
    banks = [bank for (_config, bank, _slr, _name) in compute_unit_list]
    # Each Config records the target .json it was parsed from; the ProcessPool
    # worker re-Configs from that path per CU.
    config_paths = [c.target_descriptor_path for c in configs]
    num_instances = len(configs)
    ol = build_pynq_overlay(args.xclbin)
    cu_names = enumerate_cu_names(ol)
    if len(cu_names) != num_instances:
        raise RuntimeError(f"{num_instances} configs given but overlay has "
                            f"{len(cu_names)} CUs {cu_names}")
    labels = cu_names if cu_names else [f"CU{i}" for i in range(num_instances)]
    design = PynqvivadoTernip(ol, cu_names)

    per_cu_specs = [(config_paths[i], configs[i].BatchSize) for i in range(num_instances)]
    unique_specs = list(set(per_cu_specs))

    bitnet_model_name = "microsoft/bitnet-b1.58-2B-4T"
    model = AutoModelForCausalLM.from_pretrained(bitnet_model_name).cuda()
    test_layer = U250_BitLinear(model.model.layers[0].self_attn.q_proj, unique_specs)

def init_parser():
    parser = argparse.ArgumentParser(
        description="performs Batched Generation"
    )

    parser.add_argument(
        "-c",
        "--config",
        default="/au250_xrt/xclbins/MaxCores_2_7B.json",
        help="sets config file"
    )

    parser.add_argument(
        "-m",
        "--model",
        default="MMfreeLM-370M",
        help="sets the model"
    )

    parser.add_argument(
        "-x",
        "--xclbin",
        default="/au250_xrt/xclbins/MaxCores_2_7B.xclbin",
        help="sets the model"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
