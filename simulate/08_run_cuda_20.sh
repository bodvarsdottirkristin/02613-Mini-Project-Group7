#!/bin/bash
#BSUB -J 08_cuda_kernel_20
#BSUB -q gpul40s
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o 08_cuda_kernel_20.%J.out
#BSUB -e 08_cuda_kernel_20.%J.err
#BSUB -B

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

START=$SECONDS
python 08_cuda_kernel.py 20
echo "Elapsed time: $((SECONDS - START)) seconds"
