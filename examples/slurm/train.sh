#!/bin/bash
user=$(whoami)
timestamp=$(date +%Y%m%d_%H%M%S)
work_dir=/workspace-vast/$user/exp/$(date +%Y%m%d)_slurm_test
venv_dir=/workspace-vast/$user/git/venvs/venv_axolotl
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config=$SCRIPT_DIR/meta-llama-8b-fsdp.yaml
env_file=/workspace-vast/$user/.env
huggingface_home=/workspace-vast/pretrained_ckpts

# GPU and memory configuration
num_gpus=4
mem_per_gpu=200  # GB per GPU, leaves ~400GB overhead on 2TB nodes
total_mem=$((num_gpus * mem_per_gpu))

# Ensure the experiment directory and logs directory exist
mkdir -p $work_dir/logs

# ensure HF_TOKEN and WANDB_TOKEN are in the .env file
if ! grep -q "HF_TOKEN" $env_file; then
  echo "Error: HF_TOKEN and/or WANDB_TOKEN not found in $env_file"
  exit 1
fi

cp $env_file $work_dir/.env

cat <<EOL > $work_dir/train.qsh
#!/bin/bash
#SBATCH --job-name=8B_ft
#SBATCH --output=$work_dir/logs/8B_ft_${timestamp}.out
#SBATCH --error=$work_dir/logs/8B_ft_${timestamp}.err
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --partition=high
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${total_mem}G
#SBATCH --chdir=$work_dir


export \$(grep -v '^#' $work_dir/.env | xargs)
export HF_HOME=$huggingface_home
export ACCELERATE_MAIN_PROCESS_PORT=0
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