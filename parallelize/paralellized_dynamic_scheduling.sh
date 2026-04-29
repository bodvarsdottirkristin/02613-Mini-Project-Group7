#!/bin/bash
#BSUB -J logs/parallelized_2026_dynamic
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 7
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o logs/parallelized_2026_dynamic.%J.out
#BSUB -e logs/parallelized_2026_dynamic.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python paralellized_dynamic_scheduling.py 100