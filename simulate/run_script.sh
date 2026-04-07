#!/bin/bash
#BSUB -J simulate_original
#BSUB -q hpc
#BSUB -W 00:02
#BSUB -n 4
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o simulate_original.%J.out
#BSUB -e simulate_original.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

START=$SECONDS
python simulate_original.py 20
echo "Elapsed time: $((SECONDS - START)) seconds"