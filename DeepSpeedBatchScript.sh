#!/bin/bash
#SBATCH -J deepspeed_multiNode         # Job name
#SBATCH -o deepspeed_multiNode.o%j     # Output file
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                      # Total number of tasks 
#SBATCH --ntasks-per-node=1        # Tasks per node (GPUs per node in this context)
#SBATCH --partition=gpu-a100-dev
#SBATCH -t 02:00:00                # Time limit
#SBATCH -A ASC23013         # Allocation name

# Directory to store the host file
WORKDIR=/work/09823/wnp23/ls6/project

HOSTFILE=$WORKDIR/hostfile

# Get the master node address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# Create the host file
scontrol show hostnames "$SLURM_JOB_NODELIST" > $HOSTFILE

# Append slots information to each line in the host file
sed -i "s/$/ slots=$SLURM_NTASKS_PER_NODE/" $HOSTFILE

echo "Created host file at $HOSTFILE with the following content:"
cat $HOSTFILE

echo "MASTER_ADDR=$MASTER_ADDR"
source /work/09823/wnp23/ls6/testEnv/bin/activate
module load cuda
export HF_HOME='/work/09823/wnp23/ls6/project/hf'
export PATH=$WORK/pdsh2/pdsh/bin:$PATH
export LD_LIBRARY_PATH=$WORK/pdsh2/pdsh/lib:$LD_LIBRARY_PATH

echo "Running on 1 node 1 GPU"
echo "Running with tensor parallelism"

deepspeed --num_gpus 1 --num_nodes 1 --master_addr $MASTER_ADDR --hostfile $HOSTFILE gpt2InferencePerformance.py

echo "Inference job completed."
