#!/usr/bin/env python3

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import time
import pandas as pd

# The model being hosted by your swarm
model_name = "bigscience/bloomz-560m"

# The actual initial peer from VM1 (check your latest log)
initial_peer = "/ip4/10.0.0.4/tcp/31330/p2p/12D3KooWFe3M7ws5iNmrQZAS81mptwZxzkyQMRhim2432Sz6ep6U"
no_Vms = "2"
df = pd.read_csv('prompts_tokens_9.csv')


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Connect to your swarm
model = AutoDistributedModelForCausalLM.from_pretrained(
    model_name,
    initial_peers=[initial_peer]
)
# Create a prompt to generate text
prompt = "what is the capital of India:"
RT = []
for row in df.itertuples(index=False):
    prompt = row.prompt
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    # Generate text using your distributed swarm of VMs
    outputs = model.generate(inputs, max_new_tokens=10)
    end_time = time.time()
    RT.append((end_time-start_time)*1000)

df["Responce Time"] = RT
df["No of VMs"] = [no_Vms]*len(RT)

results = pd.read_csv("Results_RT.csv")

results = pd.concat([results, df], axis=0, ignore_index=True)

results.to_csv("Results_RT.csv", index = False)


print("--- Prompt ---")
print(prompt)
print("\n--- Generation ---")
print(tokenizer.decode(outputs[0]))
