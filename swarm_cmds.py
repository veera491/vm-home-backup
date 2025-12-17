#!/usr/bin/env python3
"""
swarm_cmds.py
Generate Petals run_server commands for a private swarm with contiguous block ranges.

Usage:
  python3 swarm_cmds.py \
    --peer_id 12D3KooW... \
    --vms 10.0.0.14 10.0.0.5 10.0.0.6 \
    --model bigscience/bloomz-560m

Notes:
- VM1 creates swarm (--new_swarm)
- Others join via --initial_peers pointing to VM1
- block_indices are 0-based, end-exclusive
"""

import argparse

def partition_blocks(total_blocks: int, num_vms: int):
    q, r = divmod(total_blocks, num_vms)
    counts = [q + 1] * r + [q] * (num_vms - r)

    ranges = []
    start = 0
    for c in counts:
        end = start + c
        ranges.append((start, end))
        start = end
    return counts, ranges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer_id", required=True)
    ap.add_argument("--vms", nargs="+", required=True, help="VM IPs in execution order (VM1..VMN)")
    ap.add_argument("--model", default="bigscience/bloomz-560m")
    ap.add_argument("--port", default="31330")
    ap.add_argument("--total_blocks", type=int, default=24)
    args = ap.parse_args()

    vms = args.vms
    counts, ranges = partition_blocks(args.total_blocks, len(vms))

    vm1_ip = vms[0]
    vm1_multiaddr = f"/ip4/{vm1_ip}/tcp/{args.port}"

    print("# Block counts:", counts)
    print("# Block ranges:", ranges)
    print()

    # VM1 (bootstrap)
    s, e = ranges[0]
    print(f"""python3 -m petals.cli.run_server {args.model} \\
  --port {args.port} \\
  --public_name "{vm1_multiaddr}" \\
  --block_indices {s}:{e} \\
  --new_swarm
""")

    # VM2..VMN (join)
    for ip, (s, e) in zip(vms[1:], ranges[1:]):
        print(f"""python3 -m petals.cli.run_server {args.model} \\
  --port {args.port} \\
  --public_name "/ip4/{ip}/tcp/{args.port}" \\
  --initial_peers "{vm1_multiaddr}/p2p/{args.peer_id}" \\
  --block_indices {s}:{e}
""")

if __name__ == "__main__":
    main()
