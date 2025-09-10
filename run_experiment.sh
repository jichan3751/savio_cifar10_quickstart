#!/bin/bash

# stops script if error.
set -e

## loads conda - needed for accessing 'conda' command in slurm job.
source "$HOME/miniconda3/etc/profile.d/conda.sh"

## making sure that this script is run in 'savio_cifar10_quickstart' directory
if [ "$(basename $PWD)" != "savio_cifar10_quickstart" ]; then
    echo ERROR: please run this script in 'savio_cifar10_quickstart' directory.
    exit 1
fi

## Following conda env will be used.
ENV_PATH="./env_cifar_pt"
conda activate $ENV_PATH

## Run cifar10 training experiment
python3 main.py \
    --lr 0.1 \
    --num_train_epochs 3 \
    --output_dir "outputs/first_try_seed42" \
    --seed 42 

conda deactivate
