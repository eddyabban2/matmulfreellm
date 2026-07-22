import torch
from transformers import BitNetConfig, BitNetForCausalLM, AutoTokenizer
import transformers.integrations.bitnet as bitnet
from accelerate import infer_auto_device_map, dispatch_model
import bitnet as local_bitnet
import nvtx
import gc
from scaled_bitnet import scaled_model_config

bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear


model = BitNetForCausalLM(scaled_model_config)
model = model.to(torch.bfloat16)
total_params = sum(p.numel() for p in model.parameters())

# Calculate only trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model.total_params = total_params
model.trainable_params = trainable_params

bitnet.replace_with_bitnet_linear(
    model, 
    quantization_config=scaled_model_config.quantization_config
)

torch.save(model.state_dict(), 'scaled_bitnet.pth')