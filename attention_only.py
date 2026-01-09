#!/opt/miniconda3/envs/dejavu/bin/python
# on H100 server python is stored at /opt/miniconda3/envs/dejavu/bin/python
# attention module we are calling is stored in:
# -  /home/victoryang00/pytorch/third_party/flash-attention/flash_attn/ops/triton/layer_norm.py
# - attempting to call class LayerNormLinearFn(torch.autograd.Function):
#from torch.third_party.flash_attention.flash_attn.ops.triton.layer_norm import LayerNormLinearFn
import torch

from mmfreelm.ops.fusedbitnet import layer_norm_linear_quant_fn

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"   # must be CUDA for Triton kernels
device_name = torch.cuda.get_device_name(torch.cuda.current_device())

# Example sizes
B = 2       # batch
D = 64      # feature dim (must be reasonably small for the demo)
O = 64      # output dim for the linear layer

# Create inputs on device (float32 or float16). Triton kernels expect CUDA tensors.
x = torch.randn(B, D, device=device, dtype=torch.float32)

# Layer norm params (per-feature)
norm_weight = torch.ones(D, device=device, dtype=torch.float32)
norm_bias = torch.zeros(D, device=device, dtype=torch.float32)

# Linear layer params
linear_weight = torch.randn(O, D, device=device, dtype=torch.float32)
linear_bias = torch.zeros(O, device=device, dtype=torch.float32)

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

print("ran successfully")