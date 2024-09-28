# Accelerating LLM Inference on HPC Systems

## Overview
This repository contains the source code and scripts used for a project that explores the acceleration of large language model (LLM) inference using two advanced parallel processing frameworks: DeepSpeed and Distributed Data Parallel (DDP). The primary focus of the project is to optimize the performance of the GPT-2 model across various configurations of nodes and GPUs in high-performance computing (HPC) environments.

## Project Description
The study compares the performance of DeepSpeed and DDP for LLM inference, analyzing key metrics such as throughput, latency, power draw, GPU and memory utilization, speedup, and parallel efficiency. The findings highlight that DeepSpeed is more efficient for smaller setups with better memory management, while DDP scales better for larger datasets and higher GPU counts.

## Repository Contents
- **`ddp.py`**: Python script for running GPT-2 inference using Distributed Data Parallel (DDP) on a single node with multiple GPUs.
- **`gpt2InferencePerformanceBaseline.py`**: Baseline performance evaluation script for GPT-2 inference without parallel processing.
- **`DeepspeedGPT2InferencePerformance.py`**: Script for running GPT-2 inference using DeepSpeed with tensor parallelism.
- **`batch_ddp.sh`**: SLURM batch script for running the DDP inference job on an HPC cluster. Configured to run on the GPU partition specified in the script.
- **`DeepSpeedBatchScript.sh`**: SLURM batch script for running DeepSpeed inference on an HPC cluster with tensor parallelism.

## Experimental Setup
The experiments were conducted on the Texas Advanced Computing Center’s (TACC) "Lonestar6" system, equipped with NVIDIA A100 GPUs. The study used various configurations to assess the scalability and efficiency of DeepSpeed and DDP for LLM inference tasks.

## Results
- **DeepSpeed**: Demonstrated superior memory utilization and efficiency for small-scale setups.
- **DDP**: Showed better scalability and higher throughput for large-scale, multi-node configurations.
- **Overall Findings**: The choice between DeepSpeed and DDP should be based on the specific application requirements, such as memory constraints or the need for high throughput.

## How to Run the Code
1. **Setup Environment**: 
    - Ensure that you have the required environment set up with PyTorch, DeepSpeed, and necessary libraries installed.
    - Activate your virtual environment before running any scripts.

2. **Running with DDP**:
    - Submit the `batch_ddp.sh` script to the SLURM scheduler using the command:
      ```bash
      sbatch batch_ddp.sh
      ```
    - This script runs the `ddp.py` file using a single node and multiple GPUs.

3. **Running with DeepSpeed**:
    - Submit the `DeepSpeedBatchScript.sh` script to the SLURM scheduler using the command:
      ```bash
      sbatch DeepSpeedBatchScript.sh
      ```
    - This script runs the `DeepspeedGPT2InferencePerformance.py` file with tensor parallelism enabled.

## Future Work
- Explore additional configurations and optimizations to further improve inference efficiency.
- Experiment with larger models and more complex parallelism techniques to assess scalability in broader contexts.

## References
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Texas Advanced Computing Center (TACC)](https://www.tacc.utexas.edu/)

## Acknowledgments
This project was conducted at the Texas State University’s Department of Computer Science with the support of the Texas Advanced Computing Center (TACC).
