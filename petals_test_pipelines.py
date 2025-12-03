import asyncio
import time
from statistics import mean
from petals import DistributedBloomForCausalLM
from transformers import AutoTokenizer
import random

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_NAME = "bigscience/bloomz-560m"
INITIAL_PEERS = [
    "10.0.0.1:5001",
    "10.0.0.2:5001",
    "10.0.0.3:5001",
    "10.0.0.4:5001",
    "10.0.0.5:5001",
]

NUM_PROMPTS = 30
BATCH_SIZE = 6            # main batch size
MICROBATCH_SIZE = 2       # microbatch size within batch
PIPELINE_STAGES = 4       # for pipeline parallel

PROMPT_POOL = [
    "Explain quantum tunneling simply.",
    "What is the capital of Japan?",
    "Describe a black hole.",
    "Write a short poem.",
    "What is machine learning?",
    "Explain transformer networks.",
    "Define entropy in physics.",
]

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------

def get_prompts(n):
    return [random.choice(PROMPT_POOL) for _ in range(n)]

async def gen_one(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    await model.generate_async(inputs, max_new_tokens=20, do_sample=False)

# ------------------------------------------------------
# SEQUENTIAL MODE
# ------------------------------------------------------

async def run_sequential(model, tokenizer, prompts):
    times_per_prompt = []
    start_total = time.time()
    for p in prompts:
        t0 = time.time()
        await gen_one(model, tokenizer, p)
        times_per_prompt.append(time.time() - t0)
    total_time = time.time() - start_total
    throughput = [1/t for t in times_per_prompt]
    return total_time, times_per_prompt, throughput

# ------------------------------------------------------
# PIPELINE PARALLEL MODE
# ------------------------------------------------------

async def run_pipeline(model, tokenizer, prompts):
    times_per_prompt = []
    start_total = time.time()

    # Split prompts into batches
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        t0 = time.time()
        tasks = [asyncio.create_task(gen_one(model, tokenizer, p)) for p in batch]
        await asyncio.gather(*tasks)
        batch_time = time.time() - t0
        # approx per-prompt time in this batch
        times_per_prompt.extend([batch_time/len(batch)]*len(batch))
    total_time = time.time() - start_total
    throughput = [1/t for t in times_per_prompt]
    return total_time, times_per_prompt, throughput

# ------------------------------------------------------
# MICROBATCH MODE
# ------------------------------------------------------

async def run_microbatch(model, tokenizer, prompts):
    times_per_prompt = []
    times_per_batch = []
    times_per_microbatch = []

    start_total = time.time()
    # Main batches
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        t_batch0 = time.time()
        # Split into microbatches
        for j in range(0, len(batch), MICROBATCH_SIZE):
            microbatch = batch[j:j+MICROBATCH_SIZE]
            t_micro0 = time.time()
            # Run all prompts in microbatch concurrently
            tasks = [asyncio.create_task(gen_one(model, tokenizer, p)) for p in microbatch]
            await asyncio.gather(*tasks)
            t_micro = time.time() - t_micro0
            times_per_microbatch.append(t_micro)
            times_per_prompt.extend([t_micro/len(microbatch)]*len(microbatch))
        t_batch = time.time() - t_batch0
        times_per_batch.append(t_batch)
    total_time = time.time() - start_total
    throughput_per_prompt = [1/t for t in times_per_prompt]
    throughput_per_batch = [len(batch)/t for batch, t in zip([prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)], times_per_batch)]
    throughput_per_microbatch = [MICROBATCH_SIZE/t for t in times_per_microbatch]
    return total_time, times_per_prompt, throughput_per_prompt, times_per_batch, throughput_per_batch, times_per_microbatch, throughput_per_microbatch

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

async def main():
    prompts = get_prompts(NUM_PROMPTS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = {}

    print("\n🟦 SEQUENTIAL MODE")
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, max_rpc_concurrency=1
    )
    total_time, times_per_prompt, throughput_per_prompt = await run_sequential(model, tokenizer, prompts)
    results["Sequential"] = (total_time, times_per_prompt, throughput_per_prompt)

    print("\n🟩 PIPELINE PARALLEL MODE")
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, max_rpc_concurrency=PIPELINE_STAGES
    )
    total_time, times_per_prompt, throughput_per_prompt = await run_pipeline(model, tokenizer, prompts)
    results["Pipeline Parallel"] = (total_time, times_per_prompt, throughput_per_prompt)

    print("\n🟧 MICROBATCH MODE")
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, initial_peers=INITIAL_PEERS, max_rpc_concurrency=PIPELINE_STAGES
    )
    res = await run_microbatch(model, tokenizer, prompts)
    total_time, times_per_prompt, throughput_per_prompt, times_per_batch, throughput_per_batch, times_per_microbatch, throughput_per_microbatch = res
    results["Microbatch"] = res

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------
    print("\n\n==================== PERFORMANCE SUMMARY ====================")
    for mode, vals in results.items():
        if mode != "Microbatch":
            total_time, times_per_prompt, throughput_per_prompt = vals
            print(f"{mode} | Total Time: {total_time:.2f}s | Avg Throughput/Prompt: {mean(throughput_per_prompt):.2f} prompts/s")
        else:
            total_time, times_per_prompt, throughput_per_prompt, times_per_batch, throughput_per_batch, times_per_microbatch, throughput_per_microbatch = vals
            print(f"{mode} | Total Time: {total_time:.2f}s | Avg Throughput/Prompt: {mean(throughput_per_prompt):.2f} prompts/s")
            print(f"    Avg Time/Batch: {mean(times_per_batch):.2f}s | Avg Throughput/Batch: {mean(throughput_per_batch):.2f} prompts/s")
            print(f"    Avg Time/Microbatch: {mean(times_per_microbatch):.2f}s | Avg Throughput/Microbatch: {mean(throughput_per_microbatch):.2f} prompts/s")
    print("==============================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
