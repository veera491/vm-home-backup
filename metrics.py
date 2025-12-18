#!/usr/bin/env python3
"""
metrics.py (FINAL - interactive inputs, fixed VM IPs, cache-safe, temp cleanup)

Inputs:
  1) number of VMs (1..10)
  2) initial peer multiaddr (e.g., /ip4/10.0.0.14/tcp/31330/p2p/<peerid>)

Runs 3 modes using your thesis definitions:
  - sequential: K=1 in-flight
  - pipeline:   K=BATCH_SIZE in-flight (conveyor-belt)
  - microbatch: K=MICROBATCH in-flight (regulated conveyor-belt)

Preserves old fetchers:
  start_agents() / stop_agents() / dump_agents(clear=1&gpu=0)

Adds progress reporting.

Fixes:
  - HF cache folder not writable / missing -> uses TRANSFORMERS_CACHE or ./hf_cache and creates dirs.
  - Removes unwanted temp files after run (e.g., *.tmp).
  - Reduces memory churn by loading model once across modes.



export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home
python3 metrics.py


"""

import os
import time
import csv
import json
import gc
import asyncio
import shutil
import requests
import pandas as pd
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Optional torch cleanup (safe if torch is installed)
try:
    import torch
except Exception:
    torch = None

# ---------------- Fixed VM agent endpoints ----------------
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

# ---------------- Config ----------------
MODEL = os.environ.get("MODEL_ID", "bigscience/bloomz-560m")
PROMPTS_CSV = os.environ.get("PROMPTS_CSV", "final_prompts_data_sample.csv")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "10"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "6"))      # PP occupancy K
MICROBATCH = int(os.environ.get("MICROBATCH", "2"))      # MB occupancy K

PIPELINE_METHODS = ["sequential", "pipeline", "microbatch"]

TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "5"))
INCLUDE_VM0 = os.environ.get("INCLUDE_VM0_METRICS", "1") == "1"

QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "256"))
WRITER_FLUSH_EVERY = int(os.environ.get("WRITER_FLUSH_EVERY", "200"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "100"))

OUT_ROOT = os.environ.get("OUT_ROOT", "experiments")

PRINT_EACH_PROMPT = os.environ.get("PRINT_EACH_PROMPT", "0") == "1"

# Cache-safe defaults (fixes your /home/veera/.cache/... crash)
DEFAULT_HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.abspath("./hf_cache"))
# Also set HF_HOME to keep hub metadata under the same writable tree
DEFAULT_HF_HOME = os.environ.get("HF_HOME", os.path.abspath("./hf_home"))


# ---------------- Cache + cleanup helpers ----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def configure_hf_cache():
    """
    Ensure Hugging Face cache directories exist and are writable before any from_pretrained().
    Fixes errors like:
      "There was a problem when trying to write in your cache folder ..."
      FileNotFoundError: ... '/home/veera/.cache/huggingface/hub'
    """
    # Force to known-writable locations
    os.environ["TRANSFORMERS_CACHE"] = DEFAULT_HF_CACHE
    os.environ["HF_HOME"] = DEFAULT_HF_HOME

    # Common hub variables used by huggingface_hub
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    ensure_dir(DEFAULT_HF_CACHE)
    ensure_dir(DEFAULT_HF_HOME)

    # Also proactively create legacy default paths if they are referenced anywhere
    # (harmless if unused)
    try:
        ensure_dir(os.path.expanduser("~/.cache/huggingface/hub"))
    except Exception:
        pass

def cleanup_python_memory():
    """
    Aggressive but safe cleanup to reduce RAM growth between modes.
    """
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def remove_temp_files(paths):
    """
    Remove known temporary files (e.g., CSV tmp used during attach_vm_metrics).
    """
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ---------------- Agent helpers ----------------
def vm_ids(num_vms: int):
    ids = []
    if INCLUDE_VM0:
        ids.append("VM0")
    for i in range(1, num_vms + 1):
        vid = f"VM{i}"
        if vid in AGENTS:
            ids.append(vid)
    return ids

def start_agents(num_vms: int):
    for vid in vm_ids(num_vms):
        url = AGENTS.get(vid)
        if not url:
            continue
        try:
            requests.post(f"{url}/start", timeout=TIMEOUT)
        except Exception:
            pass

def stop_agents(num_vms: int):
    for vid in vm_ids(num_vms):
        url = AGENTS.get(vid)
        if not url:
            continue
        try:
            requests.post(f"{url}/stop", timeout=TIMEOUT)
        except Exception:
            pass

def dump_agents(num_vms: int):
    out = {}
    for vid in vm_ids(num_vms):
        url = AGENTS.get(vid)
        if not url:
            out[vid] = {}
            continue
        try:
            r = requests.get(f"{url}/dump?clear=1&gpu=0", timeout=TIMEOUT)
            out[vid] = r.json()
        except Exception:
            out[vid] = {}
    return out


# ---------------- CSV helpers ----------------
def ensure_header(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


# ---------------- Prompt iterator ----------------
def iter_prompts(df: pd.DataFrame):
    cols = set(df.columns)
    has_uid = "uid" in cols
    has_dataset = "dataset" in cols
    has_group = "group_id" in cols
    has_len = "num_tokens" in cols

    for i, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        uid = row["uid"] if has_uid else i
        dataset = row["dataset"] if has_dataset else "NA"
        group_id = int(row["group_id"]) if has_group else 0
        tok_len = int(row["num_tokens"]) if has_len else -1
        yield uid, dataset, group_id, prompt, tok_len


# ---------------- Core execution ----------------
async def gen_one(model, tokenizer, prompt: str):
    """
    One request. Returns:
      latency_ms, input_tokens, gen_tokens, throughput_tok_s, components_json
    """
    t_start = time.perf_counter()

    t_tok_s = time.perf_counter()
    enc = tokenizer(prompt, return_tensors="pt")
    inputs = enc["input_ids"]
    attn = enc.get("attention_mask", None)
    input_tokens = int(inputs.shape[-1])
    t_tok_e = time.perf_counter()

    t_gen_s = time.perf_counter()
    outputs = await asyncio.to_thread(
        model.generate,
        inputs,
        attention_mask=attn,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    t_gen_e = time.perf_counter()

    total_tokens = int(outputs.shape[-1])
    gen_tokens = max(0, total_tokens - input_tokens)

    t_end = time.perf_counter()
    latency_s = max(1e-9, t_end - t_start)
    latency_ms = round(latency_s * 1000, 2)
    throughput_tok_s = round(total_tokens / latency_s, 3)

    components = {
        "components_ms": {
            "tokenize_ms": round((t_tok_e - t_tok_s) * 1000, 3),
            "generate_ms": round((t_gen_e - t_gen_s) * 1000, 3),
            "total_ms": latency_ms,
        }
    }

    # Encourage faster memory release on CPU-only runs
    del enc, inputs, outputs
    cleanup_python_memory()

    return latency_ms, input_tokens, gen_tokens, throughput_tok_s, json.dumps(components, separators=(",", ":"))


async def run_mode(model, tokenizer, df, total_prompts: int,
                   num_vms: int, mode: str, out_csv: str, fields, vm_fields):
    """
    Distinct modes by in-flight limit K:
      sequential -> K=1
      pipeline   -> K=BATCH_SIZE
      microbatch -> K=MICROBATCH
    Uses bounded queue + K workers (conveyor-belt semantics).
    """
    if mode == "sequential":
        K = 1
    elif mode == "pipeline":
        K = BATCH_SIZE
    elif mode == "microbatch":
        K = MICROBATCH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ensure_header(out_csv, fields)

    q = asyncio.Queue(maxsize=QUEUE_MAX)
    stop_token = object()

    written = 0
    started = time.time()

    # Open writer once
    f = open(out_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)

    async def producer():
        idx = 0
        for (uid, dataset, group_id, prompt, tok_len) in iter_prompts(df):
            idx += 1
            await q.put((idx, uid, dataset, group_id, prompt, tok_len))
        for _ in range(K):
            await q.put(stop_token)

    async def worker():
        nonlocal written
        while True:
            item = await q.get()
            try:
                if item is stop_token:
                    return

                idx, uid, dataset, group_id, prompt, tok_len = item

                latency_ms, input_tokens, gen_tokens, thr_tok_s, rt_json = await gen_one(model, tokenizer, prompt)

                zero_based = idx - 1
                if mode == "sequential":
                    batch_id, micro_id, prompt_index = 0, 0, idx
                elif mode == "pipeline":
                    batch_id = zero_based // BATCH_SIZE
                    micro_id = ""
                    prompt_index = zero_based % BATCH_SIZE
                else:  # microbatch
                    batch_id = zero_based // BATCH_SIZE
                    within = zero_based % BATCH_SIZE
                    micro_id = within // MICROBATCH
                    prompt_index = within % MICROBATCH

                row = {
                    "uid": uid,
                    "dataset": dataset,
                    "group_id": group_id,
                    "Model": MODEL,
                    "Pipeline_Method": mode,
                    "No_Of_VMs": num_vms,
                    "Prompt": prompt,
                    "Token_Length": tok_len,
                    "Input_Tokens": input_tokens,
                    "Generated_Tokens": gen_tokens,
                    "Batch_ID": batch_id,
                    "MicroBatch_ID": micro_id,
                    "Prompt_Index": prompt_index,
                    "Response_Time_ms": latency_ms,
                    "Throughput_tok_s": thr_tok_s,
                    "Response_Time_Components": rt_json,
                }

                for vf in vm_fields:
                    row[vf] = ""  # filled after agent dump

                writer.writerow(row)
                written += 1

                if WRITER_FLUSH_EVERY > 0 and (written % WRITER_FLUSH_EVERY == 0):
                    f.flush()

                if PRINT_EACH_PROMPT:
                    print(f"[DONE] mode={mode} VMs={num_vms} prompt={written}/{total_prompts} rt_ms={latency_ms}")

                if (written % PROGRESS_EVERY == 0) or (written == total_prompts):
                    elapsed = time.time() - started
                    pct = (written / max(1, total_prompts)) * 100.0
                    rate = written / max(elapsed, 1e-6)
                    eta = (total_prompts - written) / max(rate, 1e-6)
                    print(
                        f"[PROGRESS] mode={mode} | VMs={num_vms} | "
                        f"{written}/{total_prompts} ({pct:.1f}%) | "
                        f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s | K={K}"
                    )
            finally:
                q.task_done()

    try:
        await asyncio.gather(producer(), *[worker() for _ in range(K)])
        f.flush()
    finally:
        try:
            f.close()
        except Exception:
            pass
        cleanup_python_memory()


def attach_vm_metrics(out_csv: str, fields, vids, agent_stats):
    tmp = out_csv + ".tmp"
    try:
        with open(out_csv, "r", newline="") as src, open(tmp, "w", newline="") as dst:
            r = csv.DictReader(src)
            w = csv.DictWriter(dst, fieldnames=fields)
            w.writeheader()
            for row in r:
                for v in vids:
                    row[f"{v}_metrics"] = json.dumps(agent_stats.get(v, {}), separators=(",", ":"))
                w.writerow(row)
        os.replace(tmp, out_csv)
    finally:
        # If interrupted, ensure tmp does not accumulate
        remove_temp_files([tmp])

def write_agent_metrics_json(out_dir: str, num_vms: int, mode: str, agent_stats: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"AgentMetrics_VM_{num_vms}_{mode}.json")
    with open(path, "w") as f:
        json.dump(agent_stats, f, separators=(",", ":"))
    return path


async def main():
    configure_hf_cache()

    print("\n=== Interactive run (fixed VM IPs, cache-safe) ===")
    print(f"MODEL={MODEL}")
    print(f"PROMPTS_CSV={PROMPTS_CSV}")
    print(f"BATCH_SIZE={BATCH_SIZE} (PP K), MICROBATCH={MICROBATCH} (MB K)")
    print(f"PROGRESS_EVERY={PROGRESS_EVERY}, QUEUE_MAX={QUEUE_MAX}")
    print(f"INCLUDE_VM0_METRICS={INCLUDE_VM0}")
    print(f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}\n")

    num_vms = int(input("Enter number of VMs to use (1-10): ").strip())
    if not (1 <= num_vms <= 10):
        raise SystemExit("VM count must be between 1 and 10.")

    peer = input("Enter initial peer multiaddr (VM1 swarm peer): ").strip()
    if not peer.startswith("/ip4/"):
        raise SystemExit("Peer must look like /ip4/<ip>/tcp/31330/p2p/<peerid>")

    df = pd.read_csv(PROMPTS_CSV)     #==========================================================================================================

    total_prompts = sum(1 for _ in iter_prompts(df))
    if total_prompts == 0:
        raise SystemExit("No prompts found in CSV (column name should be 'prompt').")

    # Tokenizer load after cache is configured
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Petals model ONCE across modes to reduce memory churn
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL, initial_peers=[peer])

    try:
        for mode in PIPELINE_METHODS:
            out_dir = os.path.join(OUT_ROOT, mode)
            out_csv = os.path.join(out_dir, f"Results_VM_{num_vms}.csv")

            vids = vm_ids(num_vms)
            vm_fields = [f"{v}_metrics" for v in vids]

            fields = [
                "uid", "dataset", "group_id",
                "Model", "Pipeline_Method", "No_Of_VMs",
                "Prompt", "Token_Length",
                "Input_Tokens", "Generated_Tokens",
                "Batch_ID", "MicroBatch_ID", "Prompt_Index",
                "Response_Time_ms", "Throughput_tok_s",
                "Response_Time_Components",
            ] + vm_fields

            print(f"\n--- START VMs={num_vms} | MODE={mode} | total_prompts={total_prompts} ---")
            print(f"peer={peer[:90]}...")
            print(f"out={out_csv}")

            tmp_files_to_remove = [out_csv + ".tmp"]

            try:
                start_agents(num_vms)
                t0 = time.time()

                await run_mode(model, tokenizer, df, total_prompts, num_vms, mode, out_csv, fields, vm_fields)

                stop_agents(num_vms)
                #agent_stats = dump_agents(num_vms)
                #attach_vm_metrics(out_csv, fields, vids, agent_stats)
                agent_stats = dump_agents(num_vms)
                metrics_path = write_agent_metrics_json(out_dir, num_vms, mode, agent_stats)
                print(f"[INFO] Agent metrics saved: {metrics_path}")


                print(f"--- DONE VMs={num_vms} MODE={mode} | wall={time.time()-t0:.1f}s ---")

            finally:
                # Always clean mode-level temp files
                remove_temp_files(tmp_files_to_remove)
                cleanup_python_memory()

        print("\n=== Completed all 3 modes ===\n")

    finally:
        try:
            del model
        except Exception:
            pass
        cleanup_python_memory()


if __name__ == "__main__":
    asyncio.run(main())

