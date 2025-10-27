#!/bin/bash

python3 -m petals.cli.run_server bigscience/bloom-3b \
  --port 31330 \
  --public_name "/ip4/10.0.0.4/tcp/31330" \
  --block_indices 0:24 \
  --new_swarm
