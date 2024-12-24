#!/bin/bash
set -eou pipefail

input_file_path=./experiments/examples/241223_running_examples/direct_request.jsonl
output_dir=./exp/pair
model="gpt-4o-mini"
temperature=1.0


python -m examples.red_teaming.pair \
    --request_input_file $input_file_path \
    --output_dir $output_dir \
    --n_steps 2 \
    --limit 3
