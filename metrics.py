#!/usr/bin/env python3
import time, requests, csv, statistics
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

MODEL_NAME = "bigscience/bloomz-560m"
PEERS = [
    "/ip4/10.0.0.4/tcp/31330/p2p/12D3KooWBjz3JRSxxqkND3TmfTqt4DRL8SKjYPhNbvevk8>
    "/ip4/10.0.0.5/tcp/31330/p2p/12D3KooWPcVVtEgoTGGdMs1q5Xc5XzD9bAysmUPk3Gmju2>
    "/ip4/10.0.0.6/tcp/33429/p2p/12D3KooWLX16Us3Z7h3QTMXc6UUfXKXqGgEckp8rgA2JR6>
]
AGENTS = {
    "VM1": "http://10.0.0.4:5001",
    "VM2": "http://10.0.0.5:5001",
    "VM3": "http://10.0.0.6:5001",
    "VM3": "http://10.0.0.6:5001",
}
PROMPT = "What is the capital of India?"
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

def run_inference(peers):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, initial>
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
    end = time.time()
    latency = (end-start)*1000
    throughput = outputs.shape[-1] / (end-start)
    return latency, throughput, outputs.shape[-1], tokenizer.decode(outputs[0])

def summarize_metrics(data):
    summary = {}
    for vm, series in data.items():
        if not series: continue
        summary[vm] = {}
        keys = [k for k in series[0].keys() if k not in ["timestamp","hostname","gpus"]]
        for k in keys:
            vals = [s[k] for s in series if k in s]
            if vals:
                summary[vm][k] = {
                    "avg": round(statistics.mean(vals),2),
                    "min": min(vals),
                    "max": max(vals)
                }
    return summary

if __name__ == "__main__":
    print("=== Distributed Monitoring System ===")
    subsets = {
        "VM1 only": PEERS[:1],
        "VM1+VM2": PEERS[:2],
        "VM1+VM2+VM3": PEERS
    }
    results = {}
    for label, peers in subsets.items():
        print(f"\n--- {label} ---")
        start_agents()
        latency, throughput, tokens, output = run_inference(peers)
        stop_agents()
        metrics = dump_agents()
        stats = summarize_metrics(metrics)
        results[label] = {
            "latency": latency,
            "throughput": throughput,
            "tokens": tokens,
            "output": output,
            "stats": stats
        }
        # Save per-run metrics
        with open("metrics_log.csv","a",newline="") as f:
            w = csv.writer(f)
            w.writerow([label,"latency",latency])
            w.writerow([label,"throughput",throughput])
            w.writerow([label,"tokens",tokens])
            for vm, vmstats in stats.items():
                for metric, vals in vmstats.items():
                    w.writerow([label,vm,metric,vals["avg"],vals["min"],vals["max"]])
        print(f"Latency {latency:.2f} ms, Throughput {throughput:.2f}, Output: {output}")
