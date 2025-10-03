#!/usr/bin/env python3

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# The model being hosted by your swarm
model_name = "bigscience/bloomz-560m"

# The actual initial peer from VM1 (check your latest log)
initial_peer = "/ip4/10.0.0.8/tcp/31330/p2p/12D3KooWKNjvy7gbmfTNmji5DrKUBiMsqUd51SeFbiTs83Go2ZeX"
print("started")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Connect to your swarm
model = AutoDistributedModelForCausalLM.from_pretrained(
    model_name,
    initial_peers=[initial_peer]
)
print("At Prompt")
# Create a prompt to generate text
prompt = "what is the capital of India:"
inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
print("At Generation")
# Generate text using your distributed swarm of VMs
outputs = model.generate(inputs, max_new_tokens=10)

print("--- Prompt ---")
print(prompt)
print("\n--- Generation ---")
print(tokenizer.decode(outputs[0]))
