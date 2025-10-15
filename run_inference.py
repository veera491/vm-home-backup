#!/usr/bin/env python3
import os

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import time
import pandas as pd

# The model being hosted by your swarm
model_name = "bigscience/bloomz-560m"

# The actual initial peer from VM1 (check your latest log)
initial_peer = "/ip4/10.0.0.4/tcp/31330/p2p/12D3KooWGMGfembFb6fSSR1UPW4A15Yo5GK6LXf3ff8FjZusuSj8"
no_Vms = "1-6"
df = pd.read_csv('prompts.csv')


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
c = 0
for row in df.itertuples(index=False):
    prompt = row.prompt
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    # Generate text using your distributed swarm of VMs
    outputs = model.generate(inputs, max_new_tokens=10)
    end_time = time.time()
    RT.append((end_time-start_time)*1000)
    print(c)
    c += 1

df["Responce Time"] = RT
df["No of VMs"] = [no_Vms]*len(RT)

filename = "Results_RT.csv"

if os.path.exists(filename):
    results = pd.read_csv(filename)
else:
    columns = ["country", "prompt", "word_count", "token_count", "Response Time", "No of VMs"]
    results = pd.DataFrame(columns=columns)


results = pd.concat([results, df], axis=0, ignore_index=True)
#df.to_csv(no_Vms+"_VM_Results.csv", index = False)
results.to_csv("Results_RT.csv", index = False)


print("--- Finished  ---")
