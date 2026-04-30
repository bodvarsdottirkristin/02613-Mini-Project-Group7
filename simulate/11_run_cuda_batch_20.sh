#!/bin/bash
#BSUB -J 11_cuda_batch_20
#BSUB -q gpul40s
#BSUB -W 03:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/11_cuda_batch_20.%J.out
#BSUB -e logs/11_cuda_batch_20.%J.err
#BSUB -B

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

cd /zhome/92/f/223286/02613-mini-project/simulate

START=$SECONDS
python 11_cuda_batch.py 20
echo "Elapsed time: $((SECONDS - START)) seconds"
