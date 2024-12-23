#!/bin/bash
set -eou pipefail

input_file_path=./experiments/examples/241223_running_examples/direct_request.jsonl
models="gpt-4o-mini"
temperature=1.0

# note that overall N = num_concurrent_k * n_steps
num_concurrent_k=10
n_steps=8
n_requests=5
request_ids="31 69 58 83 4 72 151 14 81 38" # This is a set of easy ids for quick replication, for full set use $(seq 0 158)

for model in $models; do

    model_str=${model//\//-}

    # run bon jailbreak for each specific id
    for choose_specific_id in $request_ids; do

        output_dir=./exp/bon/${model_str}/${choose_specific_id}

        if [ -f $output_dir/done_$n_steps ]; then
            echo "Skipping $output_dir because it is already done"
            continue
        fi

        python -m examples.red_teaming.bon_jailbreaking \
            --input_file_path $input_file_path \
            --output_dir $output_dir \
            --enable_cache False \
            --lm_model $model \
            --lm_temperature $temperature \
            --choose_specific_id $choose_specific_id \
            --num_concurrent_k $num_concurrent_k \
            --n_steps $n_steps
    done
done
