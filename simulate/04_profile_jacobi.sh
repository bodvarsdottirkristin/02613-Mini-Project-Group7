#!/bin/bash
#BSUB -J simulate_original
#BSUB -q hpc
#BSUB -W 00:04
#BSUB -n 4
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o logs/04_profile_jacobi.%J.out
#BSUB -e logs/04_profile_jacobi.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

kernprof -l simulator.py
python -m line_profiler -rmt logs/simulator.py.lprof