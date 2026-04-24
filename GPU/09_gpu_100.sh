#!/bin/bash
#BSUB -J 09_gpu_100
#BSUB -q hpc
#BSUB -W 00:04
#BSUB -n 4
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o logs/09_gpu_100.%J.out
#BSUB -e logs/09_gpu_100.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

START=$SECONDS
python 09_gpu.py 100
echo "Elapsed time: $((SECONDS - START)) seconds"