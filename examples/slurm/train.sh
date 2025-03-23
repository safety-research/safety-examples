#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
work_dir=/workspace/exp/johnh/250323_slurm_test
venv_dir=/home/johnh/git/axolotl/.venv
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config=$SCRIPT_DIR/qlora-fsdp-8b-test.yaml

# Ensure the experiment directory and logs directory exist
mkdir -p $work_dir/logs

cat <<EOL > $work_dir/train.qsh
#!/bin/bash
#SBATCH --job-name=8B_ft
#SBATCH --output=$work_dir/logs/8B_ft_${timestamp}.out
#SBATCH --error=$work_dir/logs/8B_ft_${timestamp}.err
#SBATCH --gres=gpu:8
#SBATCH --partition=gpupart

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