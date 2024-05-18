#!/bin/bash -l

#SBATCH -o out/%x_%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 128
#SBATCH -p batch
#SBATCH --time=0-06:00:00
#SBATCH --qos=normal
#SBATCH --mail-type=all
#SBATCH --mail-user=thibault.simonetto@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
# export LOGLEVEL="ERROR"
# export PYTHONWARNINGS="ignore"
# export MODELS_DIR="/scratch/users/tsimonetto/drift-study/models2"
# export MODELS_DIR="/scratch/users/tsimonetto/drift-study/models"
conda activate crobustkdd

echo "SCRIPT: $@"

eval "$@"
