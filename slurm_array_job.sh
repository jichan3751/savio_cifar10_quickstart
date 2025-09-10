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

#SBATCH --time=02:00:00 # time requested (D-HH:MM:SS or HH:MM:SS)

## stdout/err: output logs and error logs goes to here.
#SBATCH --output=slurm_logs/slurm-%A_%a.out.txt
#SBATCH --error=slurm_logs/slurm-%A_%a.err.txt

## array options:
#SBATCH --array=0-3%2  # array index = 0,1,2,3 will be running, with at most 2 jobs running in parallel.

echo "Running SLURM job: Host $HOSTNAME, job id: $SLURM_ARRAY_JOB_ID array task id: $SLURM_ARRAY_TASK_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

bash run_experiment_array.sh
