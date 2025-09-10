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

## conda environment will be created in following path.
ENV_PATH="./env_cifar_pt"

## remove current environment if exists.
echo Installing new env at $ENV_PATH ...
[ -d $ENV_PATH ] && echo "removing $ENV_PATH ..." && rm -rf $ENV_PATH
mkdir -p $ENV_PATH

## create new conda environment
conda create -y --prefix $ENV_PATH python=3.7 pip

## install packages in conda environment
# ... do pip install, conda install here

conda activate $ENV_PATH

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install ipython ipdb matplotlib

conda deactivate

