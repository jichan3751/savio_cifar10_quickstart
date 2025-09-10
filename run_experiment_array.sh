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

JOB_ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

if [ "$JOB_ARRAY_INDEX" == 0 ]; then
    LR=0.1
    SEED=42
elif [ "$JOB_ARRAY_INDEX" == 1 ]; then
    LR=0.1
    SEED=43
elif [ "$JOB_ARRAY_INDEX" == 2 ]; then
    LR=0.01
    SEED=42
elif [ "$JOB_ARRAY_INDEX" == 3 ]; then
    LR=0.01
    SEED=43
else
    echo JOB_ARRAY_INDEX = "$JOB_ARRAY_INDEX" is not defined!
    exit 1
fi

## Run cifar10 training experiment

OUTPUT_DIR="outputs/lr${LR}_seed${SEED}"

python3 main.py \
    --lr $LR \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --seed $SEED 
