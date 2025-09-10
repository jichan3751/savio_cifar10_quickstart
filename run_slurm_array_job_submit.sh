#!/bin/bash
set -e

if [ -d "outputs" ]; then
    echo "reset outputs directory"
    rm -r outputs
fi
mkdir -p outputs

if [ -d "slurm_logs" ]; then
    echo "reset slurm logs"
    rm -r slurm_logs
fi
mkdir -p slurm_logs

## submits the array job script
sbatch slurm_array_job.sh
