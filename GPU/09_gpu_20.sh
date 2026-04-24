#!/bin/bash
#BSUB -J 09_gpu_20
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/09_gpu_20.%J.out
#BSUB -e logs/09_gpu_20.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

START=$SECONDS
python 09_gpu.py 20
echo "Elapsed time: $((SECONDS - START)) seconds"