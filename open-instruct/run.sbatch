#!/bin/bash
#SBATCH -J llama-debug
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=/home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/train_debug_%j.log
#SBATCH --error=/home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/train_debug_%j.log
export PYTHONPATH=/home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct:$PYTHONPATH

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"

# -----------------------------
# Conda
# -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-training-env


# -----------------------------
# CUDA Setup - Use conda environment as CUDA_HOME
# Now that cuda-toolkit is installed, this will work properly
# -----------------------------
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib64:$CUDA_HOME/lib

# -----------------------------
# REQUIRED on this cluster
# These prevent DeepSpeed from trying to compile anything
# -----------------------------
export DS_BUILD_OPS=0
export DS_BUILD_UTILS=0
export DS_BUILD_CPU_ADAM=0
export DS_SKIP_CUDA_CHECK=1

# -----------------------------
# Environment
# -----------------------------
export HF_HOME=/home/compiling-ganesh/24m0829/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: improve CPU data loading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -----------------------------
# Sanity checks
# -----------------------------
nvidia-smi
lscpu | head -n 20
echo "CUDA_HOME set to: $CUDA_HOME"
which nvcc
nvcc --version

# -----------------------------
# Run training
# -----------------------------
cd /home/compiling-ganesh/24m0829/language-task-arithmetic/open-instruct/scripts
bash custom_train.sh

echo "Job finished at $(date)"
