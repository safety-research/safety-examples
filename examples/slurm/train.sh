#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
work_dir=/workspace/exp/johnh/250323_slurm_test
venv_dir=/home/johnh/git/axolotl/.venv
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config=$SCRIPT_DIR/qlora-fsdp-8b-test.yaml
env_file=$SCRIPT_DIR/../../.env
huggingface_home=/workspace/pretrained_ckpts

# Ensure the experiment directory and logs directory exist
mkdir -p $work_dir/logs

# ensure HF_TOKEN and WANDB_TOKEN are in the .env file
if ! grep -q "HF_TOKEN" $env_file || ! grep -q "WANDB_TOKEN" $env_file; then
  echo "Error: HF_TOKEN and/or WANDB_TOKEN not found in $env_file"
  exit 1
fi

cp $env_file $work_dir/.env

cat <<EOL > $work_dir/train.qsh
#!/bin/bash
#SBATCH --job-name=8B_ft
#SBATCH --output=$work_dir/logs/8B_ft_${timestamp}.out
#SBATCH --error=$work_dir/logs/8B_ft_${timestamp}.err
#SBATCH --gres=gpu:8
#SBATCH --partition=gpupart

export \$(grep -v '^#' $work_dir/.env | xargs)
export HF_HOME=$huggingface_home
cd $work_dir
source $venv_dir/bin/activate
axolotl preprocess $config
axolotl train $config
EOL

# submit job
sbatch $work_dir/train.qsh

echo "To see the queue, run:"
echo "watch squeue"

# echo the log file
echo "To view the log file, run:"
echo "tail -f $work_dir/logs/8B_ft_${timestamp}.out"