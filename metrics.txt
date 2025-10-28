#!/usr/bin/env python3
import os
import time, requests, csv, statistics
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import pandas as pd
import json

MODEL = "bigscience/bloomz-560m"
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
        try:
            requests.post(f"{url}/start", timeout=2)
        except:
            pass


def stop_agents():
    for vm, url in AGENTS.items():
        try:
            requests.post(f"{url}/stop", timeout=2)
        except:
            pass


def dump_agents():
    data = {}
    for vm, url in AGENTS.items():
        try:
            data[vm] = requests.get(f"{url}/dump", timeout=5).json()
        except:
            data[vm] = []
    return data


def run_inference(MODEL, PROMPT):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL, initial_peers=[PEERS])
    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
    end = time.time()
    Run_Time_ = (end - start) * 1000
    throughput_ = outputs.shape[-1] / Run_Time_
    return Run_Time_, throughput_


if __name__ == "__main__":
    print("=== Distributed Monitoring System ===")

    df = pd.read_csv('prompts.csv')

    c = 1
    RT = []
    throughputs = []
    VM1, VM2, VM3, VM4, VM5, VM6, VM7, VM8, VM9, VM10 = [], [], [], [], [], [], [], [], [], []
    for row in df.itertuples(index=False):
        prompt = row.prompt
        start_agents()
        Run_Time, throughput = run_inference(MODEL, prompt)
        stop_agents()

        print("================Done Prompt id", c, "Run time", Run_Time, "===================")

        metrics = dump_agents()

        RT.append(Run_Time)
        throughputs.append(throughput)

        with open("metrics.jsonl", "a") as f:
            json.dump(metrics, f)
            f.write("\n")

        c += 1

    with open("metrics.jsonl") as f:
        for line in f:
            metrics = json.loads(line)

            VM1.append(metrics["VM1"])
            VM2.append(metrics["VM2"])
            VM3.append(metrics["VM3"])
            VM4.append(metrics["VM4"])
            VM5.append(metrics["VM5"])
            VM6.append(metrics["VM6"])
            VM7.append(metrics["VM7"])
            VM8.append(metrics["VM8"])
            VM9.append(metrics["VM9"])
            VM10.append(metrics["VM10"])

    n = len(RT)
    results_df = pd.DataFrame(colums=["Model", "No.Of. VMs", "Prompt", "Token Length", "Response Time", "Throughput",
                                      "VM1", "VM2," "VM3", "VM4", "VM5", "VM6", "VM7", "VM8", "VM9", "VM10"])
    results_df["Model"] = [MODEL] * n
    results_df["No.Of. VMs"] = [No_of_VMs] * n
    results_df["Prompt"] = df["Prompt"]
    results_df["Token Length"] = df["Token Length"]
    results_df["Response Time"] = RT
    results_df["Throughput"] = throughputs

    results_df["VM1"] = VM1
    results_df["VM2"] = VM2
    results_df["VM3"] = VM3
    results_df["VM4"] = VM4
    results_df["VM5"] = VM5
    results_df["VM6"] = VM6
    results_df["VM7"] = VM7
    results_df["VM8"] = VM8
    results_df["VM9"] = VM9
    results_df["VM10"] = VM10

    filename = "Results_VM_1_to_10.csv"
    if os.path.exists(filename):
        results = pd.read_csv(filename)
    else:
        results = pd.DataFrame(colums=["Model", "No.Of. VMs", "Prompt", "Token Length", "Response Time", "Throughput",
                                       "VM1", "VM2," "VM3", "VM4", "VM5", "VM6", "VM7", "VM8", "VM9", "VM10"])

    results = pd.concat([results, results_df], axis=0, ignore_index=True)
    results.to_csv(filename, index=False)

    print("--- Finished  ---")
