#!/bin/bash
set -eoux pipefail

output_dir=./exp/capability_evals
model="gpt-4o-mini"
temperature=0.0


python -m examples.capability_evals.multi_choice.run_multi_choice \
    --output_dir $output_dir \
    --dataset tiny_mmlu \
    --path_to_dataset $output_dir/tiny_mmlu.jsonl \
    --path_to_output $output_dir/tiny_mmlu_${model}.jsonl \
    --model $model \
    --temperature $temperature \
    --n_rows_to_process 3
