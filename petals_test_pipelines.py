#!/usr/bin/env python3
import asyncio
import time
import random
from statistics import mean

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM


MODEL_NAME = "bigscience/bloomz-560m"

# your real peer
INITIAL_PEERS = [
    "/ip4/10.0.0.14/tcp/31330/p2p/12D3KooWFJzmKZyK7BFEVN7UWYjA26xsZpsCf1xZcY4yCbJHCiuA"
]

NUM_PROMPTS = 30
BATCH_SIZE = 6
MICROBATCH = 2


PROMPT_POOL = [
    "Explain quantum tunneling simply.",
    "What is the capital of Japan?",
    "Describe a black hole.",
    "Write a short poem.",
    "What is machine learning?",
    "Explain transformer networks.",
    "Define entropy in physics.",
]


def get_prompts(n):
    return [random.choice(PROMPT_POOL) for _ in range(n)]


async def gen_one(model, tokenizer, prompt):
    """Thread-safe wrapper around model.generate."""
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    return await asyncio.to_thread(
        model.generate,
        inputs,
        max_new_tokens=20,    # REQUIRED (cannot use max_length simultaneously)
        do_sample=False
    )


async def run_sequential(model, tokenizer, prompts):
    times = []
    t0 = time.time()

    for p in prompts:
        s = time.time()
        await gen_one(model, tokenizer, p)
        times.append(time.time() - s)

    total = time.time() - t0
    throughput = [1/t for t in times]
    return total, times, throughput


async def run_pipeline(model, tokenizer, prompts):
    times = []
    t0 = time.time()

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        s = time.time()

        tasks = [asyncio.create_task(gen_one(model, tokenizer, p)) for p in batch]
        await asyncio.gather(*tasks)

        b = (time.time() - s) / len(batch)
        times.extend([b] * len(batch))

    total = time.time() - t0
    throughput = [1/t for t in times]
    return total, times, throughput


async def run_microbatch(model, tokenizer, prompts):
    times_prompt = []
    times_batch = []
    times_micro = []

    t0 = time.time()

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        bs = time.time()

        for j in range(0, len(batch), MICROBATCH):
            micro = batch[j:j+MICROBATCH]
            ms = time.time()

            tasks = [asyncio.create_task(gen_one(model, tokenizer, p)) for p in micro]
            await asyncio.gather(*tasks)

            m = time.time() - ms
            times_micro.append(m)
            times_prompt.extend([m/len(micro)] * len(micro))

        b = time.time() - bs
        times_batch.append(b)

    total = time.time() - t0
    throughput_prompt = [1/t for t in times_prompt]
    throughput_batch = [MICROBATCH/t for t in times_batch]
    throughput_micro = [MICROBATCH/t for t in times_micro]

    return (total, times_prompt, throughput_prompt,
            times_batch, throughput_batch,
            times_micro, throughput_micro)


async def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = get_prompts(NUM_PROMPTS)
    results = {}

    print("\n🟦 SEQUENTIAL MODE")
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    results["Sequential"] = await run_sequential(model, tokenizer, prompts)
    del model

    print("\n🟩 PIPELINE MODE")
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    results["Pipeline"] = await run_pipeline(model, tokenizer, prompts)
    del model

    print("\n🟧 MICROBATCH MODE")
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS)
    results["Microbatch"] = await run_microbatch(model, tokenizer, prompts)
    del model

    print("\n============== SUMMARY ==============")
    for mode, vals in results.items():
        if mode != "Microbatch":
            total, tpp, thpp = vals
            print(f"{mode}: Total={total:.2f}s  Avg Throughput={mean(thpp):.2f} gen/s")
        else:
            (total, t_p, th_p, t_b, th_b, t_m, th_m) = vals
            print(f"{mode}: Total={total:.2f}s  Avg Throughput={mean(th_p):.2f} gen/s")
            print(f"  Batch Throughput={mean(th_b):.2f}, Micro Throughput={mean(th_m):.2f}")
    print("======================================\n")


if __name__ == "__main__":
    asyncio.run(main())
