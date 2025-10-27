#!/usr/bin/env python3
import time, requests, csv, statistics
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

MODEL_NAME = ["bigscience/bloom-3b"]
PEERS = input("Enter The Initial Peer: ")
No_of_VMs = int(input("Enter No of VMs: "))
AGENTS = {
    "VM1": "http://10.0.0.4:5001",
    "VM2": "http://10.0.0.5:5001",
    "VM3": "http://10.0.0.6:5001",
    "VM4": "http://10.0.0.7:5001",
    "VM5": "http://10.0.0.8:5001",
    "VM6": "http://10.0.0.9:5001",
    "VM7": "http://10.0.0.10:5001",
    "VM8": "http://10.0.0.11:5001",
    "VM9": "http://10.0.0.12:5001",
    "VM10": "http://10.0.0.13:5001",
}
PROMPTS = ["What is the capital of India?"]
MAX_NEW_TOKENS = 10

def start_agents():
    for vm, url in AGENTS.items():
        try: requests.post(f"{url}/start", timeout=2)
        except: pass

def stop_agents():
    for vm, url in AGENTS.items():
        try: requests.post(f"{url}/stop", timeout=2)
        except: pass

def dump_agents():
    data = {}
    for vm, url in AGENTS.items():
        try: data[vm] = requests.get(f"{url}/dump", timeout=5).json()
        except: data[vm] = []
    return data

def run_inference(MODEL, PROMPT):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL, initial_peers=[PEERS]
    )
    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
    end = time.time()
    Run_Time = (end-start)*1000
    throughput = outputs.shape[-1] / Run_Time
    return Run_Time, throughput

if __name__ == "__main__":
    print("=== Distributed Monitoring System ===")

    results = []

    for MODEL in MODEL_NAME:
        for PROMPT in PROMPTS:
            start_agents()
            Run_Time, throughput = run_inference(MODEL, PROMPT)
            stop_agents()
            metrics = dump_agents()

            print(MODEL, PROMPT, No_of_VMs, throughput, Run_Time, metrics)
