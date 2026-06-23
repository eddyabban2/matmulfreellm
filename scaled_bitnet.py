import torch
from transformers import BitNetConfig, BitNetForCausalLM, AutoTokenizer
import transformers.integrations.bitnet as bitnet
from accelerate import infer_auto_device_map, dispatch_model
import bitnet as local_bitnet
import nvtx
import gc

# 1. Apply your custom BitNet monkey patches
bitnet.pack_weights = local_bitnet.pack_weights
bitnet.unpack_weights = local_bitnet.unpack_weights
bitnet.BitLinear = local_bitnet.BitLinear
bitnet._replace_with_bitnet_linear = local_bitnet._replace_with_bitnet_linear
bitnet.replace_with_bitnet_linear = local_bitnet.replace_with_bitnet_linear

class quant_config: 
    quant_method = "bitnet"
    linear_class = "autobitlinear"
    quantization_mode = "offline"
    modules_to_not_convert = []
    def to_dict(self): 
        return {  
            "quant_method": self.quant_method,
            "linear_class":  self.linear_class,
            "quantization_mode": self.quantization_mode,
            "modules_to_not_convert" :[]
        }
quantization_config = quant_config()

standard_model_config = BitNetConfig(
    bos_token_id= 128000,
    eos_token_id= 128001,
    hidden_act="relu2",
    hidden_size=2560,           
    initializer_range=0.02,
    intermediate_size=6912,     
    max_position_embeddings=4096,
    model_type="bitnet",
    num_attention_heads=20,     
    num_hidden_layers=30,       
    num_key_value_heads=5,
    rms_norm_eps=1e-05,
    rope_theta=500000.0,
    torch_dtype="bfloat16",
    tie_word_embeddings=True,
    use_cache=True,            
    vocab_size=128256,
    quantization_config=quantization_config
)

def create_custom_bitnet(model_config=standard_model_config):
    model = BitNetForCausalLM(model_config)
    model = model.to(torch.bfloat16)

    bitnet.replace_with_bitnet_linear(
        model, 
        quantization_config=model_config.quantization_config
    )
    model.cuda()
    return model

def main():
    model_config = standard_model_config
    model_config.num_hidden_layers = 50
    model = create_custom_bitnet(model_config=model_config)
    prompts = [
        "Explain the concept of quantum computing.",
        "Write a short story about a space explorer.",
        "What is the capital of France?",
        "How does BitNet work?", 
        "Eddy was here"
    ]

    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T") 
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
    gc.collect()
    torch.cuda.empty_cache()
    memory_usage = torch.cuda.memory_allocated()
    print(f"Memory Usage: {memory_usage/(1024**3)} GiB") 

    print(f"Processing batch of {len(prompts)} prompts...")

    # Inference
    with nvtx.annotate("workload", color="cyan"):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

    # Decode results
    for i, output in enumerate(outputs):
        print(f"\n--- Output {i+1} ---\n{tokenizer.decode(output, skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
