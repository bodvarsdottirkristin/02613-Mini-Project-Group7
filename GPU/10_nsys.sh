#!/bin/bash
#BSUB -J 10_nsys
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]" 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/10_nsys.%J.out
#BSUB -e logs/10_nsys.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

nsys profile -o 10_nsys_prof python 09_gpu.py 10