#!/bin/bash
#BSUB -J parallelized
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 7
#BSUB -R "rusage[mem=512MB] span[hosts=1] select[model==XeonGold6126]"
#BSUB -o parallelized.%J.out
#BSUB -e parallelized.%J.err
#BSUB -B 

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python paralellized.py 100