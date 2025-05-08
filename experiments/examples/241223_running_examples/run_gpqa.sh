#!/bin/bash
set -eoux pipefail

output_dir=./exp/capability_evals
model="gpt-4o-mini"
temperature=0.0


echo python -m examples.capability_evals.multi_choice.run_multi_choice \
    --output_dir $output_dir \
    --dataset gpqa \
    --path_to_dataset $output_dir/gpqa.jsonl \
    --path_to_output $output_dir/gpqa_${model}.jsonl \
    --model $model \
    --temperature $temperature
