#!/usr/bin/env python3
"""
Optimized single-file Petals client benchmark (CPU-only) with your naming:
  - SEQUENTIAL MODE
  - PIPELINE MODE
  - MICROBATCH MODE

Design goals:
- Minimal overhead (single model load, pre-tokenize prompts once)
- Correct latency + throughput (req/s, gen_tokens/s) and tail latency (P95)
- Queue / conveyor-belt semantics for PIPELINE + MICROBATCH (global in-flight limit)

Note (thesis wording):
- Petals is pipeline-parallel internally for every request.
- PIPELINE/MICROBATCH here are client-side workload shaping modes (different in-flight limits).
"""

import asyncio
import os
import time
import random
import concurrent.futures
from statistics import mean

import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# -------------------- Config --------------------
MODEL_NAME = "bigscience/bloomz-560m"
INITIAL_PEERS = [
    "/ip4/10.0.0.14/tcp/31330/p2p/12D3KooWKTKKWt5Jp7dPRZcFKemGPDupEJywztgpVQJSuEJHvKjB"
]

NUM_PROMPTS = 30
BATCH_SIZE = 6      # PIPELINE MODE occupancy (K)
MICROBATCH = 2      # MICROBATCH MODE occupancy (K)
MAX_NEW_TOKENS = 20
SEED = 1234

# Thread pool size for to_thread work; keep modest to avoid oversubscription.
# Defaults to max(8, 2*CPU) but you can override via env var.
DEFAULT_WORKERS = max(8, (os.cpu_count() or 4) * 2)
THREAD_WORKERS = int(os.getenv("PETALS_CLIENT_WORKERS", str(DEFAULT_WORKERS)))

PROMPT_POOL = [
    "Explain quantum tunneling simply.",
    "What is the capital of Japan?",
    "Describe a black hole.",
    "Write a short poem.",
    "What is machine learning?",
    "Explain transformer networks.",
    "Define entropy in physics.",
]

# -------------------- Helpers --------------------
def get_prompts(n: int, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(PROMPT_POOL) for _ in range(n)]

def p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(0, int(0.95 * (len(xs) - 1)))
    return xs[k]

def summarize(mode_name: str, wall: float, lats: list[float], gen_tokens: list[int]):
    n = len(lats)
    avg_lat = mean(lats) if lats else float("nan")
    req_s = (n / wall) if wall > 0 else 0.0
    gen_tok_s = (sum(gen_tokens) / wall) if wall > 0 else 0.0

    print(f"\n============== {mode_name} SUMMARY ==============")
    print(f"Total wall time: {wall:.2f}s")
    print(f"Requests: {n}")
    print(f"Avg latency: {avg_lat:.2f}s | P95 latency: {p95(lats):.2f}s")
    print(f"Throughput: {req_s:.2f} req/s")
    print(f"Generation throughput: {gen_tok_s:.2f} gen_tokens/s")
    print("=================================================\n")

# -------------------- Core execution --------------------
async def gen_one(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """
    Execute one request via model.generate() in a thread, return (latency_s, generated_tokens).
    input_ids already prepared on CPU.
    """
    in_tokens = int(input_ids.shape[-1])

    t0 = time.perf_counter()
    out_ids = await asyncio.to_thread(
        model.generate,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    lat = time.perf_counter() - t0

    total_tokens = int(out_ids.shape[-1])
    out_tokens = max(0, total_tokens - in_tokens)
    return lat, out_tokens

async def run_sequential(model, encoded_inputs):
    """True sequential: 1 in-flight at all times."""
    t0 = time.perf_counter()
    lats, outs = [], []
    for (input_ids, attention_mask) in encoded_inputs:
        lat, out_tok = await gen_one(model, input_ids, attention_mask)
        lats.append(lat)
        outs.append(out_tok)
    wall = time.perf_counter() - t0
    return wall, lats, outs

async def run_queue_mode(model, encoded_inputs, occupancy: int):
    """
    Queue / conveyor-belt at request level:
    - keep up to `occupancy` requests in-flight
    - refill immediately when any request finishes
    """
    sem = asyncio.Semaphore(occupancy)

    async def worker(inp):
        async with sem:
            input_ids, attention_mask = inp
            return await gen_one(model, input_ids, attention_mask)

    t0 = time.perf_counter()
    tasks = [asyncio.create_task(worker(inp)) for inp in encoded_inputs]

    lats, outs = [], []
    for done in asyncio.as_completed(tasks):
        lat, out_tok = await done
        lats.append(lat)
        outs.append(out_tok)

    wall = time.perf_counter() - t0
    return wall, lats, outs

# -------------------- Main --------------------
async def main():
    print(f"Client thread workers: {THREAD_WORKERS} (override with PETALS_CLIENT_WORKERS)")
    prompts = get_prompts(NUM_PROMPTS)

    # Tokenizer: set pad token once (helps if you later batch-tokenize).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize prompts ONCE to remove tokenizer overhead from mode comparisons.
    # Keep tensors on CPU.
    encoded_inputs = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt")
        encoded_inputs.append((enc["input_ids"], enc.get("attention_mask", None)))

    # One model load for all modes (more stable and faster).
    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL_NAME,
        initial_peers=INITIAL_PEERS,
    )

    # Control Python threadpool size used by asyncio.to_thread
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_WORKERS))

    # ---------------- SEQUENTIAL MODE ----------------
    print("\n🟦 SEQUENTIAL MODE")
    wall, lats, outs = await run_sequential(model, encoded_inputs)
    summarize("SEQUENTIAL MODE", wall, lats, outs)

    # ---------------- PIPELINE MODE ----------------
    print(f"\n🟩 PIPELINE MODE (occupancy={BATCH_SIZE})")
    wall, lats, outs = await run_queue_mode(model, encoded_inputs, occupancy=BATCH_SIZE)
    summarize("PIPELINE MODE", wall, lats, outs)

    # ---------------- MICROBATCH MODE ----------------
    print(f"\n🟧 MICROBATCH MODE (occupancy={MICROBATCH})")
    wall, lats, outs = await run_queue_mode(model, encoded_inputs, occupancy=MICROBATCH)
    summarize("MICROBATCH MODE", wall, lats, outs)

if __name__ == "__main__":
    asyncio.run(main())
