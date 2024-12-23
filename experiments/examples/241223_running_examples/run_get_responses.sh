#!/bin/bash
set -eou pipefail

input_file_path=./experiments/examples/241223_running_examples/direct_request.jsonl
output_dir=./exp/get_responses_and_classification
model="gpt-4o-mini"
temperature=1.0


python -m examples.inference.get_responses \
    --input_file $input_file_path \
    --output_dir $output_dir \
    --model_ids $model \
    --temperature $temperature \
    --request_tag "behavior_str" \
    --file_name "responses.jsonl"


python -m examples.inference.run_classifier \
    --response_input_file $output_dir/responses.jsonl \
    --output_dir $output_dir \
    --classifier_models $model \
    --temperature $temperature \
    --file_name "classifier-responses.jsonl"
