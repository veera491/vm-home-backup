#!/usr/bin/env python3
import os, time, csv, json, requests, sys
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import pandas as pd

# ---------- Config ----------
MODEL = os.environ.get("MODEL_ID", "bigscience/bloomz-560m")
PEER = input("Enter the initial peer multiaddr: ").strip()
NUM_VMS = int(input("Enter number of VMs: ").strip())
PROMPTS_CSV = os.environ.get("PROMPTS_CSV", "prompts.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "Results_VM_1_to_10.csv")

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
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "10"))
TIMEOUT = 3  # seconds for agent http calls


# ----------------------------

def start_agents():
    for _, url in list(AGENTS.items())[:NUM_VMS]:
        try:
            requests.post(f"{url}/start", timeout=TIMEOUT)
        except:
            pass


def stop_agents():
    for _, url in list(AGENTS.items())[:NUM_VMS]:
        try:
            requests.post(f"{url}/stop", timeout=TIMEOUT)
        except:
            pass


def dump_agents():
    """Get summarized metrics and clear on agents."""
    data = {}
    for vm, url in list(AGENTS.items())[:NUM_VMS]:
        try:
            data[vm] = requests.get(f"{url}/dump?clear=1&gpu=0", timeout=TIMEOUT).json()
        except:
            data[vm] = {}
    return data


def iter_prompts(csv_path):
    # expects columns: Prompt, Token Length (Token Length optional)
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            prompt = row.get("Prompt") or row.get("prompt") or row.get("text") or ""
            tok_len = row.get("Token Length") or row.get("token_length") or ""
            yield prompt, tok_len


def ensure_header(path, fieldnames):
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def append_row(path, fieldnames, row):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


if __name__ == "__main__":
    print("=== Distributed Monitoring ===")
    # Prepare output CSV
    VM_FIELDS = [f"VM{i}_metrics" for i in range(1, NUM_VMS + 1)]
    FIELDS = ["Model", "No_Of_VMs", "Prompt", "Token_Length", "Response_Time_ms", "Throughput_tok_s"] + VM_FIELDS
    ensure_header(OUTPUT_CSV, FIELDS)

    # Load model/tokenizer ONCE
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL, initial_peers=[PEER])

    df = pd.read_csv(PROMPTS_CSV)
    total_prompts = len(df)

    try:
        for idx, (prompt, tok_len) in enumerate(iter_prompts(PROMPTS_CSV), start=1):
            prompt = (prompt or "").strip()
            if not prompt:
                continue

            # start metrics, run inference, stop metrics
            start_agents()
            inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
            dt = time.time() - t0  # seconds
            stop_agents()

            # compute metrics
            gen_tokens = int(outputs.shape[-1])
            latency_ms = round(dt * 1000.0, 2)
            throughput = round(gen_tokens / max(dt, 1e-6), 3)  # tokens / sec

            # fetch summarized agent stats (already cleared on agents)
            agent_stats = dump_agents()

            # build one compact CSV row
            row = {
                "Model": MODEL,
                "No_Of_VMs": NUM_VMS,
                "Prompt": prompt,
                "Token_Length": tok_len,
                "Response_Time_ms": latency_ms,
                "Throughput_tok_s": throughput,
            }
            # keep each VM metrics JSON compact as a string (doesn't blow up RAM)
            for i in range(1, NUM_VMS + 1):
                vm_key = f"VM{i}"
                vm_col = f"VM{i}_metrics"
                row[vm_col] = json.dumps(agent_stats.get(vm_key, {}), separators=(",", ":"))

            append_row(OUTPUT_CSV, FIELDS, row)
            print(f"[{idx}/{total_prompts}] Response Time={latency_ms} ms, thrpt={throughput} tok/s")

            # free tensors promptly
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # Explicit cleanup (esp. if CUDA)
        del model, tokenizer
        import gc;

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("--- Finished ---")
