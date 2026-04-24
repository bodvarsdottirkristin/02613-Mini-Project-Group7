#!/bin/bash
#BSUB -J 09_gpu_10_2026
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1"
#BSUB -o logs/09_gpu_10_2026.%J.out
#BSUB -e logs/09_gpu_10_2026.%J.err
#BSUB -B 


module load cuda/11.8

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026


export CUPY_NVCC_GENERATE_CODE="arch=compute_70,code=sm_70"
rm -rf ~/.cupy/kernel_cache


python -c "import cupy as cp; print(f'GPUs visible: {cp.cuda.runtime.getDeviceCount()} | Compute Capability: {cp.cuda.Device().compute_capability}')"

START=$SECONDS
python 09_gpu.py 10
echo "Elapsed time: $((SECONDS - START)) seconds"