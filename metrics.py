#!/usr/bin/env python3
import os
import time
import csv
import json
import requests
import sys

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from petals import AutoDistributedModelForCausalLM

# ---------- Config ----------
MODEL = os.environ.get("MODEL_ID", "bigscience/bloomz-560m")

NUM_VMS = int(
    input(
        "Enter number of Petals server VMs (excluding client VM0, 0 = standalone): "
    ).strip()
)

if NUM_VMS > 0:
    PEER = input("Enter the initial peer multiaddr: ").strip()
else:
    PEER = None

PROMPTS_CSV = os.environ.get("PROMPTS_CSV", "final_prompts_data_sample.csv")

# Batch / microbatch parameters (purely for indexing / analysis)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "6"))
MICROBATCH = int(os.environ.get("MICROBATCH", "2"))

PIPELINE_METHODS = ["sequential", "pipeline", "microbatch"]

# VM0 = CLIENT
AGENTS = {
    "VM0": "http://10.0.0.4:5001",
    "VM1": "http://10.0.0.14:5001",
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
TIMEOUT = 3  # seconds
# ----------------------------


def _vm_ids():
    ids = ["VM0"]
    for i in range(1, NUM_VMS + 1):
        vmid = f"VM{i}"
        if vmid in AGENTS:
            ids.append(vmid)
    return ids


def start_agents():
    for vmid in _vm_ids():
        url = AGENTS.get(vmid)
        if url:
            try:
                requests.post(f"{url}/start", timeout=TIMEOUT)
            except:
                pass


def stop_agents():
    for vmid in _vm_ids():
        url = AGENTS.get(vmid)
        if url:
            try:
                requests.post(f"{url}/stop", timeout=TIMEOUT)
            except:
                pass


def dump_agents():
    out = {}
    resp = None
    for vmid in _vm_ids():
        url = AGENTS.get(vmid)
        if not url:
            out[vmid] = {}
            continue
        try:
            resp = requests.get(f"{url}/dump?clear=1&gpu=0", timeout=TIMEOUT)
            out[vmid] = resp.json()
        except:
            out[vmid] = {}

        print("=====", vmid, out, resp)
    return out


def ensure_header(path, fields):
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def append_row(path, fields, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def iter_prompts(df):
    for row in df.itertuples(index=False):
        p = (row.prompt or "").strip()
        if not p:
            continue
        yield row.uid, row.dataset, int(row.group_id), p, row.num_tokens


# ----------------------------

if __name__ == "__main__":
    ext1 = time.time()
    print("\n=== Distributed Monitoring (3 modes in one run) ===")
    print(f"VM Count = {NUM_VMS}")
    print(f"Prompts CSV = {PROMPTS_CSV}")
    print(f"BATCH_SIZE = {BATCH_SIZE}, MICROBATCH = {MICROBATCH}\n")

    vm_ids = _vm_ids()
    VM_FIELDS = [f"{vmid}_metrics" for vmid in vm_ids]

    # Add Batch / MicroBatch indices
    FIELDS = [
        "uid", "dataset", "group_id",
        "Model", "Pipeline_Method", "No_Of_VMs",
        "Prompt", "Token_Length",
        "Input_Tokens", "Generated_Tokens",
        "Batch_ID", "MicroBatch_ID", "Prompt_Index",
        "Response_Time_ms", "Throughput_tok_s",
        "Response_Time_Components"
    ] + VM_FIELDS

    df = pd.read_csv(PROMPTS_CSV).head(5)

    total_prompts = len(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

    for PIPELINE_METHOD in PIPELINE_METHODS:
        # Output directory and file for this mode
        OUTPUT_DIR = os.path.join("experiments", PIPELINE_METHOD)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"Results_VM_{NUM_VMS}.csv")

        print(f"\n--- MODE: {PIPELINE_METHOD} ---")
        print(f"Saving to = {OUTPUT_CSV}")
        xt1 = time.time()
        ensure_header(OUTPUT_CSV, FIELDS)

        # Load model depending on VM count
        if NUM_VMS == 0:
            model = AutoModelForCausalLM.from_pretrained(MODEL)
        else:
            model = AutoDistributedModelForCausalLM.from_pretrained(
                MODEL, initial_peers=[PEER]
            )

        try:
            for idx, (uid, dataset, group_id, prompt, tok_len) in enumerate(
                iter_prompts(df), start=1
            ):

                start_agents()
                t_start = time.time()

                # Tokenization
                t_tok_s = time.time()
                encoded = tokenizer(prompt, return_tensors="pt")
                inputs = encoded["input_ids"]
                input_tokens = int(inputs.shape[-1])
                t_tok_e = time.time()

                # Sanity check: tokenization
                if input_tokens <= 0:
                    stop_agents()
                    _ = dump_agents()
                    print(f"[WARN] {PIPELINE_METHOD} uid={uid}: no tokens, skipping.")
                    continue

                # Generate
                t_gen_s = time.time()
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
                t_gen_e = time.time()

                # Sanity check: generation
                if outputs is None or outputs.shape[-1] <= 0:
                    stop_agents()
                    _ = dump_agents()
                    print(f"[WARN] {PIPELINE_METHOD} uid={uid}: empty output, skipping.")
                    continue

                stop_agents()

                # Dump metrics
                t_dump_s = time.time()
                agent_stats = dump_agents()
                t_dump_e = time.time()

                # Latency
                total_time_s = t_gen_e - t_start
                latency_ms = round(total_time_s * 1000, 2)

                # Tokens
                total_tokens = int(outputs.shape[-1])
                generated_tokens = max(0, total_tokens - input_tokens)
                throughput = round(total_tokens / max(total_time_s, 1e-6), 3)

                # Response-time components
                rt_components = {
                    "timestamps": {
                        "t_start": t_start,
                        "t_tokenize_start": t_tok_s,
                        "t_tokenize_end": t_tok_e,
                        "t_generate_start": t_gen_s,
                        "t_generate_end": t_gen_e,
                        "t_dump_start": t_dump_s,
                        "t_dump_end": t_dump_e,
                    },
                    "components_ms": {
                        "client_processing_time_ms": round((t_tok_e - t_start) * 1000, 3),
                        "model_compute_time_ms": round((t_gen_e - t_gen_s) * 1000, 3),
                        "comm_delay_ms": None,
                        "orchestration_overhead_ms": None,
                        "total_response_time_ms": latency_ms,
                    },
                }

                # Compute batch / microbatch indices (purely logical mapping)
                zero_based_idx = idx - 1

                if PIPELINE_METHOD == "sequential":
                    batch_id = 0
                    micro_id = 0
                    prompt_index = idx

                elif PIPELINE_METHOD == "pipeline":
                    batch_id = zero_based_idx // BATCH_SIZE
                    micro_id = None
                    prompt_index = zero_based_idx % BATCH_SIZE

                elif PIPELINE_METHOD == "microbatch":
                    batch_id = zero_based_idx // BATCH_SIZE
                    within_batch = zero_based_idx % BATCH_SIZE
                    micro_id = within_batch // MICROBATCH
                    prompt_index = within_batch % MICROBATCH

                else:
                    batch_id = None
                    micro_id = None
                    prompt_index = idx

                # Build row
                row = {
                    "uid": uid,
                    "dataset": dataset,
                    "group_id": group_id,
                    "Model": MODEL,
                    "Pipeline_Method": PIPELINE_METHOD,
                    "No_Of_VMs": NUM_VMS,
                    "Prompt": prompt,
                    "Token_Length": tok_len,
                    "Input_Tokens": input_tokens,
                    "Generated_Tokens": generated_tokens,
                    "Batch_ID": batch_id,
                    "MicroBatch_ID": micro_id,
                    "Prompt_Index": prompt_index,
                    "Response_Time_ms": latency_ms,
                    "Throughput_tok_s": throughput,
                    "Response_Time_Components": json.dumps(rt_components),
                }

                # Add VM metrics
                for vmid in vm_ids:
                    row[f"{vmid}_metrics"] = json.dumps(
                        agent_stats.get(vmid, {}), separators=(",", ":")
                    )

                append_row(OUTPUT_CSV, FIELDS, row)

                print(
                    f"[{PIPELINE_METHOD}] [{idx}/{total_prompts}] "
                    f"uid={uid} | batch={batch_id} micro={micro_id} "
                    f"| {latency_ms} ms | thr={throughput}"
                )

                # Free memory
                del encoded, inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"--- Finished MODE: {PIPELINE_METHOD} --- Takes Time", time.time() - xt1, "-----")
        print(f"Saved results to: {OUTPUT_CSV}\n")

    print("======Total Experiment Time", time.time() - ext1,"========")
    print("\n=== All 3 modes completed for this VM count ===\n")
