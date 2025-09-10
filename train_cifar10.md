# Running ResNet18 + CIFAR10 training
In this part of tutorial, we will learn how to submit jobs that can use GPUs to run CIFAR10 training code.
We will mainly use 1 GTX1080 11GB GPU for our training, but we'll also learn how to request for other type of GPUs.

## Contents
1. [Preliminaries](#preliminaries)
2. [Submitting interactive job](#submitting-interactive-job)
2. [Submitting non-interactive job](#submitting-non-interactive-job)
2. [How to use request other type of GPUs?](#how-to-use-request-other-type-of-gpus)
2. [Cost of using Savio resources ](#cost-of-using-savio-resources)
2. [Submitting array job](#submitting-array-job)

## Preliminaries
Before we start, download this codebase to access the code in this repository and 
make sure that your terminal's working directory is `savio_quickstart_tutorial` by doing following:

- In [our lab wiki github repository page](https://github.com/the-chen-lab/wiki), click green 'Code' button -> click 'Download ZIP'
- Upload the zip file to any directory you want, and open terminal inside that directory.
- Run following command to unzip:
```shell
unzip wiki-main.zip
```
- Move to `savio_quickstart_tutorial` directory.
```
cd wiki-main/code-data-compute/savio_cifar10_quickstart
```

## Submitting interactive job
Interactive jobs are the jobs connect to the compute node through SSH, and allow using terminal to run command, run debugging session or provide input to your program. 

In this tutorial, we consider using 1 GTX1080TI GPU with 2 CPUs.

### Checking resource availability
Lets first check if these resources are available.
Run below to check node status (please copy the whole command):
```
$ sinfo -p savio2_1080ti -O PartitionName,NodeHost,StateCompact,CPUsState,AllocMem,Memory,Gres,GresUsed,Reason
```
You'll see a tabulated output with some columns. Below are the meaning of each column:
- `STATE`: 
	- `idle`, `mix` : Able to accept job
	- `alloc` : all resources allocated and cannot accept job
	- `down`, `drain` : Not available to use.
- `CPUS(A/I/O/T)` : allocated/idle/other/total 
- `ALLOCMEM`, `MEMORY` : allocated memory, total memory (in MB)
- `GRES`, `GRES_USED` : Total GPUs / Allocated GPUs
	- Ex. `GRES` : `gpu:GTX1080TI:4`, `GRES_USED`: `gpu:GTX1080TI:1(IDX:none`
	- In this compute node, among 4 GTX1080TI GPUs, 1 GPU is used. (which means 3 GPUs are available to use).

You'll need to check if there is node(=host) that has 2 CPUs and 1 GTX1080TI available. 

After checking availability, run below command below to request one job with 1 GTX 1080TI GPU for interactive use:
```
$ srun --pty -A fc_chenlab -p savio2_1080ti --qos savio_normal --gres=gpu:GTX1080TI:1 --cpus-per-task=2 -t 02:00:00 bash
srun: job 19803207 queued and waiting for resources
srun: job 19803207 has been allocated resources
```
It normally takes around 5~10 seconds to allocate a node.
If not assigned, it is likely that your options are wrong or requested resource is not available.

After job is allocated, note that the hostname changed in your terminal: Ex. `[<username>@ln003]` -> `[<username>@n0227]`. You are now connected to `n0227` node.

Note that the above `srun` command assigns a node for only 2 hours for your use and will disconnect after 2 hours. If you want to use it for more time, change the `-t` parameter. Note that if you request for more hours, you will be less likely to get a node assigned.

- *Note for Open Ondemand code-server users*: if you run above `srun ...` command in Open Ondemand code-server, you may see following error message:
```
srun: error: Step requested GRES but job doesn't have GRES
srun: error: Unable to create step for job 20192207: Invalid generic resource (gres) specification
```
- If you see above error, you can log in to `ln002` node, `cd` to same directory and run the same `srun ...` command:
```
$ ssh ln002 -t "cd $(pwd); bash"
$ srun --pty -A fc_chenlab -p savio2_1080ti --qos savio_normal --gres=gpu:GTX1080TI:1 --cpus-per-task=2 -t 02:00:00 bash
```

Run below to check the hostname and the GPUs assigned to your session:
```shell
$  echo hostname: $(hostname), assigned GPU: $CUDA_VISIBLE_DEVICES
hostname: n0227.savio2, assigned GPU: 0
```
```shell
# monitor usage of currently assigned GPU
$ nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=index,gpu_name,memory.total,memory.used,utilization.gpu --format=csv
```

Before we run training code, we will setup conda environment to use pytorch software. Please run following:
```shell
$ bash run_environment_setup.sh
```
This script installs conda environment in `./env_cifar_pt` directory.

You can activate this environment by: `conda activate ./env_cifar_pt`.

We now run our training code:
```shell
$ bash run_experiment.sh
```
This code will download CIFAR10 dataset in `./data` and run 3 epochs of training ResNet18, and write train results in `outputs/first_try_seed42/results.json`.

I recommend reading what each script is doing: [run_environment_setup.sh](./run_environment_setup.sh), [run_experiment.sh](./run_experiment.sh).

After training is done, you can exit the compute node by running `exit` or pressing 'Ctrl + D'.

## Submitting non-interactive job 

The default mode of running computation jobs in Savio cluster is to run in *non-interactive* mode. 

In this mode, when you submit a job, the job will be queued and wait, and it will run when requested resources become available. The text output (what you see normally see in terminal) will be written in text files.

Submit the job by running following:
```
bash run_slurm_job_submit.sh
```
The job ID of submitted job will be displayed.

The scripts involved in running this command are as below. 

- [run_slurm_job_submit.sh](./run_slurm_job_submit.sh)
    - Removes previous outputs directory if exists.
    - Creates logging directory, submit the job using `sbatch` command.
    - The actual command that submits the job is `sbatch slurm_job.sh` in this script.
- [slurm_job.sh](./slurm_job.sh)
    - Contains all resource specification to request to SLURM job scheduer.
        - The lines starting with `#SBATCH` are read by `sbatch` command.
    - The bottom part of the script will be run as commands.
- [run_experiment.sh](./run_experiment.sh)
    - Our CIFAR10 training script.

I highly recommend taking a look at each above script carefully so that you can modify to your future needs.

Note that we don't have to specify memory or disk usage in `slurm_job.sh` script.These are handled automatically.

After submitting the job, you can monitor your job's status by:
```
# Simple view
$ squeue -u $USER

# To see more info
$ squeue -u $USER -o "%i|%u|%P|%q|%N|%C|%b|%T|%e|%L" 
```

If the job starts to run (with status 'RUNNING' or 'R'), you'll see that text output of your script in in `slurm_logs` directory. Note that normal outputs will be written in `slurm-%j.out.txt` and error messages are writtin in `slurm-%j.err.txt`,where `%j` is the job id.

After training is done, you'll again see your train results in `outputs/first_try_seed42/results.json`.

In case your job is not running, check if the requested resources are available by `sinfo ...` command introduced in [checking resource availability](#checking-resource-availability) section.

You may get more information on why the job is not running by running following:
```
$ sq
```

If you wish to cancel the submission of job, run `scancel` command:
```
$ scancel <job_id>
```

## How to use request other type of GPUs?

Besides GTX1080TI (VRAM 11GB) that we used in this tutorial, there are also other types of GPUs that you can use: GTX2080TI (VRAM 11GB), A5000 (VRAM 24GB), A40 (VRAM 45GB), ...

In order to use them, you'll need to figure out following 3 components:
1. Partition: 
	- Where group of computing nodes are in. 
	- Often similar types of machines are in the same partition
	- Ex. `savio2_1080ti`, `savio3_gpu`, `savio4_gpu`,..
2. QOS(Quality of Service): 
	- For each partition, there are QOSes that allows using different set of computing nodes.
	- Ex. `savio_normal` in `savio2_1080ti` partition, `a5k_gpu4_normal` in `savio4_gpu` partition, ...
3. GPUs / CPUs.
	- You'll need to figure out which GPUs are in which partition and which QOS.
	- Also there is specific ratio of CPUs/GPUs for each GPU type. **If you don't match this ratio in your submit script, your job will not run!**.

### Checking available partition and QOS
Run following command to check which QOS and partition you have access to:
```shell
$ sacctmgr -p show associations user=$USER
```
Below is part of my output:
```
brc|fc_chenlab|jichan3751|savio2_1080ti|1|||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_chenlab|jichan3751|savio3_gpu|1|||||||||||||a40_gpu3_normal,gtx2080_gpu3_normal,savio_lowprio,v100_gpu3_normal|gtx2080_gpu3_normal||
brc|fc_chenlab|jichan3751|savio4_gpu|1|||||||||||||a5k_gpu4_normal,savio_lowprio|a5k_gpu4_normal||
```
Below are explanation of what this means:
- In `savio2_1080ti` partition, you can use `savio_normal` QOS to use GTX1080TI GPUs.
- In `savio3_gpu` partition, 
    - You can use `a40_gpu3_normal` QOS to use A40 GPUs.
    - You can use `gtx2080_gpu3_normal` QOS to use GTX2080 GPUs.
    - You can use `v100_gpu3_normal` QOS to use V100 GPUs.
- In `savio4_gpu` partition
    - You can use `a5k_gpu4_normal` QOS to use A5000 GPUs.

For more information QOS and GPU type relation, check [Savio Docs: GPU job](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/#gpu-jobs).

### Checking CPUs/GPU ratio for different GPU type
You can check the ratio the table of [Savio Docs: GPU job](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/#gpu-jobs).

Some examples of ratio of CPU:GPU :
- 1080TI, GTX2080TI = 2:1
- A5000 = 4:1
- A40 = 8:1

### Example submit options
Below are some example options of requesting for GPUs.
You can apply this info in [slurm_job.sh](./slurm_job.sh).
```
## Using 1 GTX1080TI GPU in savio2_1080ti partition.
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:GTX1080TI:1 --cpus-per-task=2
#SBATCH --qos=savio_normal

## Using 2 GTX1080TI GPUs in savio2_1080ti partition. 
## Note that we request 2x CPUs when we request 2 GPUs.
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:GTX1080TI:2 --cpus-per-task=4
#SBATCH --qos=savio_normal

## Using 1 GTX2080TI GPU in savio3_gpu partition.
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:GTX2080TI:1 --cpus-per-task=2
#SBATCH --qos=gtx2080_gpu3_normal

## Using 1 A40 GPU in savio3_gpu partition.
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:1 --cpus-per-task=8
#SBATCH --qos=a40_gpu3_normal

## Using 1 A5000 GPU in savio4_gpu partition.
#SBATCH --partition=savio4_gpu
#SBATCH --gres=gpu:A5000:1 --cpus-per-task=4
#SBATCH --qos=a5k_gpu4_normal
```

Run following to check resource availability of all available GPU partitions: note the `-p` partition parameter includes all GPU partitions.
```shell
$ sinfo -p savio2_gpu,savio2_1080ti,savio3_gpu,savio4_gpu -O PartitionName,NodeHost,StateCompact,CPUsState,AllocMem,Memory,Gres,GresUsed,Reason
```

For more information on GPU job submission, read following references:
- [Savio Docs: About GPU jobs](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/#gpu-jobs)
- [Savio Docs: Example GPU job script](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/#gpu-jobs)

## Cost of using Savio resources

**Savio is not a free resource!**

When you submit a job and it runs, it consumes credits that our lab has. 

The main currency of Savio credit is SU(Service Unit)s.
To estimate SUs that is used by your job, below is the formula:

```
Consumed SUs = (# of hours used) * (# of cores) * (partition-specific scaling factor)
```
For information on partition-specific scaling factor, see [Savio Docs: Scaling of Service Units](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/service-units-savio/#scaling-of-service-units).

Example of caculating SUs:
- Use GTX1080TI GPU in `savio2_1080ti` with 2 CPUs for 3 hours:
    -  `3(hrs) * 2(cpus) * 1.67(scaling) = 10.02 SUs`
- Use A40 GPU in `savio3_gpu` with 8 CPUs for 3 hours:
    -  `3(hrs) * 8(cpus) * 3.67(scaling) = 88.08 SUs`

Generally you'll have to pay a lot more SUs if you use expensive GPUs.

You can check your SU usage by:
```
# Total usage
check_usage.sh

# To check usage during (start date ~ end date)
check_usage.sh -s YYYY-MM-DD -e YYYY-MM-DD 
```
You can also check your usage in [MyBRC portal](https://mybrc.brc.berkeley.edu/).

We normally have 300K SUs available for each year.
Please use the SU credits responsibly.

There is also ways to use the GPUs for FREE: check out [Low-priority Jobs](./low_priority_jobs.md) section.

If your job expects high SU usage (for example, needs more than 2 GPUs or expensive GPUs and/or has to run for several days), please consider submitting job as [Low-priority Jobs](./low_priority_jobs.md) to avoid overusing the credit.

## Submitting array job
Array jobs are multiple jobs that can submitted within single submission command.

Suppose you want to do hyperparameter search for the training, or run same experiments with  multiple different random seeds to make sure that your experimental results did not came out of luck.

Consider running our CIFAR10 experiment with following 4 settings:
- learning_rate = 0.1, seed = 42
- learning_rate = 0.1, seed = 43
- learning_rate = 0.01, seed = 42
- learning_rate = 0.01, seed = 43

Instead of submitting 4 jobs with 4 different scripts with different parameters, we can submit 1 script that does the same job.

The scripts involved in this task are as follows:
- [run_slurm_array_job_submit.sh](./run_slurm_array_job_submit.sh)
    - Creates logging directory, submit the job using `sbatch` command.
    - The actual command that submits the job is `sbatch slurm_array_job.sh`. 
- [slurm_array_job.sh](./slurm_array_job.sh)
    - Contains all resource specification to request to SLURM job scheduer.
        - The lines starting with `#SBATCH` are read by `sbatch` command.
    - The bottom part of the script will be run as commands.
    - Note the change in `SBATCH --array=0-3%2`:    
        - 4 jobs will be submitted, and at most 2 jobs will run in parallel.
        - Each job will run with 1 GTX 1080 GPU with 2 CPUs.
        - Environment variable `SLURM_ARRAY_TASK_ID` will be set as index of array job: 0,1,2,3 for each job. 
- [run_experiment_array.sh](./run_experiment_array.sh)
    - For each different SLURM_ARRAY_TASK_ID, the script will set the parameter of training script as well as output directory, and run the training code.

Again, please take a look at each script carefully to use it for your needs.

To submit the array job, run the following:
```
$ bash run_slurm_array_job_submit.sh
```

You can check the output/error logs for each array job in `slurm_logs` directory.


## Additional information
- The CIFAR10 training code is based on https://github.com/kuangliu/pytorch-cifar.

## Next step
Please check out ways to use the GPUs for FREE: check out [Low-priority Jobs](./low_priority_jobs.md) section.


