#!/bin/bash

python3 -m petals.cli.run_server bigscience/bloomz-560m \
  --port 31330 \
  --public_name "/ip4/10.0.0.4/tcp/31330" \
  --block_indices 0:5 \
  --new_swarm
