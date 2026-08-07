from vllm import LLM

# For generative models (task=generate) only
llm = LLM(model='ridger/MMfreeLM-2.7B', runner="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)

# # For pooling models (task={embed,classify,reward,score}) only
# llm = LLM(model=..., task="embed")  # Name or path of your model
# output = llm.encode("Hello, my name is")
# print(output)