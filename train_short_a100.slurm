#!/bin/bash
#SBATCH --job-name=train_video_gat
#SBATCH --account=ec35
#SBATCH --mail-type=FAIL
#SBATCH --partition=accel
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=20
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:1
#========================================
# #SBATCH --gpus=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --qos=devel
# #SBATCH --mem-per-gpu=40G
# #SBATCH --gpus-per-node=1
# # SBATCH --partition=accel
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module --quiet purge
module --latest load Miniconda3
# module load CUDAcore/11.2.1
module load CUDA/12.1.1

# Source the conda.sh script from the system Miniconda3 installation
source /cluster/software/EL9/easybuild/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh

# Initialize Conda for non-interactive shells
eval "$(conda shell.bash hook)"

conda activate ml-mas

# Reload the CUDA module to ensure environment variables are set
# module load CUDAcore/11.2.1
module load CUDA/12.1.1

# Debug output
echo "Loaded modules:"
module list
echo "Available conda environments:"
conda info --envs
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

echo "Training with arguments: $@"

# Show GPU information
nvidia-smi

# Run your Python script
python src/main.py "$@"

# ~/.conda/envs/ml-mas/bin/python src/main.py "$@"