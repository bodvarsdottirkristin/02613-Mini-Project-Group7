#!/bin/bash
#BSUB -J 12_cuda_batch
#BSUB -q gpul40s
#BSUB -W 03:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o 12_cuda_batch.%J.out
#BSUB -e 12_cuda_batch.%J.err
#BSUB -B

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd /zhome/92/f/223286/02613-mini-project/simulate

START=$SECONDS
python 12_cuda_batch.py
echo "Elapsed time: $((SECONDS - START)) seconds"
