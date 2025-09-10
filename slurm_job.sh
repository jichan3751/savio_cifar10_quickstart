#!/bin/bash

## Credit account options (savio)
#SBATCH --account=fc_chenlab

## Nodes, tasks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

## Resources 
## GPUs and CPUs
## CPUs: In savio, there is designated amount of CPUs 
##       that needs to be requested depending on type of GPUs.
## RAM: Normally we don't specify memory.

## Using 1 GTX1080TI GPU in savio2_1080ti partition.
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:GTX1080TI:1 --cpus-per-task=2
#SBATCH --qos=savio_normal

#SBATCH --time=02:00:00 # time requested : 3 days (HH:MM:SS)

## stdout/err: output logs and error logs goes to here.
#SBATCH --output=slurm_logs/slurm-%j.out.txt
#SBATCH --error=slurm_logs/slurm-%j.err.txt

echo "Running SLURM job: Host $HOSTNAME, job id: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

bash run_experiment.sh
