#!/bin/bash
#BSUB -J visualize
#BSUB -q hpc
#BSUB -W 00:04
#BSUB -n 4
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o logs/visualize.%J.out
#BSUB -e logs/visualize.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python visualize.py