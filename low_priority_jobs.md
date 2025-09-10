## Low-priority jobs : using GPUs for FREE

Low-priority jobs allows us to use the Savio's resource for **FREE**, without consuming SUs. 

**One caveat is : jobs may be interrupted.**

When you submit a job in low-priority mode, your job will run when resource is available. But when high-priority jobs(jobs that uses SUs, or jobs from node owner lab) are queued and there is no more available resource, your job will be killed to run their job(=pre-emptied).

After your job is pre-emptied, these job can be requeued; your job can run again when resource becomes available again.

Practically, if you submit a low-priority job that runs less than 2 hours, it is likely that it will finish eventually - it will keep getting killed and restarted, and complete when no other jobs interrupts your job. However, for longer job (> 2 hours) it is very less likely that your job will be able to finish due to interruption. 

**You can bypass this if your program implements checkpoint/resume.** Instead of restarting from scratch, the program can resume from recent saved checkpoint proceed from there. 
- For example, many pytorch training code saves checkpoint every epoch, and is able to resume training from saved checkpoint.
    - If program saved checkpoint at epoch 3 and was interrupted during training epoch 4, when job gets restarted, it can resume training from end of epoch 3.

I've been using low-priority mode for most of my GPU usage. My current usage is 0.18SUs even after using hundreds of GPUs for several months. Hope you can leverage this feature too!

## Submitting jobs to low-priorty QOS

Make sure you have access to `savio_lowprio` QOS by running following:
```shell
$ sacctmgr -p show associations user=$USER | grep savio_lowprio
```
My output shows that I can access in `savio_lowprio` QOS  `savio3_gpu` and `savio4_gpu`.
```
brc|fc_chenlab|jichan3751|savio4_gpu|1|||||||||||||a5k_gpu4_normal,savio_lowprio|a5k_gpu4_normal||
brc|fc_chenlab|jichan3751|savio3_gpu|1|||||||||||||a40_gpu3_normal,gtx2080_gpu3_normal,savio_lowprio,v100_gpu3_normal|gtx2080_gpu3_normal||
```

### Submitting jobs to low-priorty QOS: interactive mode
Below command submits interactive mode job request to use 1 A5000 and 4 CPUs for 2 hours:
```shell
$ srun --pty -A fc_chenlab -p savio4_gpu --qos savio_lowprio --gres gpu:A5000:1 --cpus-per-task 4 -t 02:00:00 bash
```

### Submitting jobs to low-priorty QOS: non-interactive mode
In order to apply it in submit script(e.g. [slurm_job.sh](./slurm_job.sh)), disable GPU-specific QOS option and use `savio_lowprio` QOS with requeue option. Below is the example of requesting 1 A5000 GPU:
```shell
#SBATCH --partition=savio4_gpu
#SBATCH --gres=gpu:A5000:1 --cpus-per-task=4
###SBATCH --qos=a5k_gpu4_normal # disabling this option

#SBATCH --qos=savio_lowprio
#SBATCH --requeue  # requeues after premption
#SBATCH --open-mode=append # appends the log outputs at re-run
```

## Responsible use of low-priority QOS
(* Below are author's personal opinions.)

Because low-priority QOS are free, we often see people overusing this resource in a way that can interfere with other people's usage. 

I suggest everyone using low-priority QOS to use it responsibly, avoid overusage, and make effort to keep high resource availability availble for everyone to use. 

Below are my suggestions to avoid resource hogging:
- For array jobs that runs longer than 2 hour, please limit maximum jobs running in parallel to 8 ~ 16.
    - You can specify this in `--array=0-127%8` (8 is number of maximum tasks that can be run in parallel)
- For array jobs that runs **shorter than 2 hours**, and resources are available to run all jobs in parallel, I think it's fine to run all process in parallel.
    - Other users should be able to find availabilty in 2 hours.
    - But if there is *less resource availability*, I suggest maximum jobs running in parallel should be again limited to 8~16.
