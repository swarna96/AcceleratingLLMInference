#!/bin/bash
#SBATCH --job-name=pytorch_distributed_inference
#SBATCH --output=job_torchgpt_inference_%j.txt
#SBATCH --error=error_torchgpt_inference_%j.txt
#SBATCH --partition=gpu-a100-dev  # Change to the appropriate GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Total number of tasks
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=ASC23013  # Make sure this is your correct account
#SBATCH --export=ALL

source torch_env/bin/activate


# Define MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(hostname)
export MASTER_PORT=57646
export NCCL_DEBUG=INFO

# Run the Python script with distributed launch

srun python ddp.py

