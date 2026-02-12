#!/opt/miniconda3/envs/dejavu/bin/python

# on H100 server python is stored at /opt/miniconda3/envs/dejavu/bin/python
# attention module we are calling is stored in:
# -  /home/victoryang00/pytorch/third_party/flash-attention/flash_attn/ops/triton/layer_norm.py
# - attempting to call class LayerNormLinearFn(torch.autograd.Function):
#from torch.third_party.flash_attention.flash_attn.ops.triton.layer_norm import LayerNormLinearFn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmfreelm.ops.fusedbitnet import layer_norm_linear_quant_fn
import argparse

parser = argparse.ArgumentParser(
    description="creates a csv file with benchmark results"
)

parser.add_argument(
    "-b", 
    "--batch_size",
    default=32,
    help="sets the sequence length of input tokens"
)

parser.add_argument(
    "-i", 
    "--iterations",
    default=10,
    help="sets the number of iterations to run attention for"
)

args = parser.parse_args()

#data_type = [ torch.float32,torch.float16,torch.bfloat16 ]
data_type = torch.float16

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"   # must be CUDA for Triton kernels
device_name = torch.cuda.get_device_name(torch.cuda.current_device())
# model_name = 'ridger/MMfreeLM-2.7B'
# model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
# dimensions = int(model.config.hidden_size)
dimensions=2560

# Example sizes
B = int(args.batch_size)       # batch
D = dimensions      # feature dim (must be reasonably small for the demo)
O = dimensions      # output dim for the linear layer

# Create inputs on device (float32 or float16). Triton kernels expect CUDA tensors.
x = torch.randn(B, D, device=device, dtype=data_type)

# Layer norm params (per-feature)
norm_weight = torch.ones(D, device=device, dtype=data_type)
norm_bias = torch.zeros(D, device=device, dtype=data_type)

# Linear layer params
linear_weight = torch.randn(O, D, device=device, dtype=data_type)
linear_bias = torch.zeros(O, device=device, dtype=data_type)

# Forward call
# is_rms_norm=True if you want to use RMSNorm behavior (the BitLinear used is_rms_norm=True)
out = layer_norm_linear_quant_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=True,
)
iter = int(args.iterations)
for _ in range(iter):
    _ = layer_norm_linear_quant_fn(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=True,
    )