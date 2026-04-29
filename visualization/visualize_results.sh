#!/bin/bash
#BSUB -J visualize_results
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -R "rusage[mem=2GB]"
#BSUB -o logs/visualize_results.%J.out
#BSUB -e logs/visualize_results.%J.err
#BSUB -B

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd /zhome/92/f/223286/02613-mini-project/visualization
python visualize_results.py
