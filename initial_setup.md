# Initial setup

## Account creation
Follow instructions in Savio Documentations:
- Create a MyBRC account : [Savio Docs: Setting up a MyBRC User Portal account](https://docs-research-it.berkeley.edu/services/high-performance-computing/getting-account/#setting-up-a-mybrc-user-portal-account)
- Get access to `fc_chenlab` project : [Getting Access to a Project](https://docs-research-it.berkeley.edu/services/high-performance-computing/getting-account/#getting-access-to-a-project)

## Logging in 
Please follow instructions in [Savio Docs: Logging into BRC Clusters](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/logging-brc-clusters/#logging-into-brc-clusters).

You'll need to setup OTP(One Time password) generator to log in. 

You can log in to savio cluster using following command:
```
$ ssh <username>@hpc.brc.berkeley.edu
```
You will be connected to one of two login servers: `ln002` or `ln003`.
If you wish to connect to a specific login server, run following command:
```
$ ssh <username>@ln002.brc.berkeley.edu 
$ ssh <username>@ln003.brc.berkeley.edu
```

When you are connected into Savio cluster, you can also use following command to ssh to specific the login server:
```
$ ssh ln002
$ ssh ln003
```


## Recommended development environment

If you are used to working with VSCode's Remote-SSH extension to work with remote compute servers, unfortunately, Savio does not allow this extension. Instead I recommend checking out Savio's Open OnDemand code-server, which will run web code editor with experience mostly identical to VSCode.

Below are some example options for development environment to use in Savio cluster:
1. (*Recommended*) Open Ondemand code-server
	- Features:
		- Similar experience to VSCode.
			- Easily upload / download / edit files from file explorer.
			- Launch integrated terminal with keyboard (Ctrl + `)
			- Launch integrated terminal in specific directory: right-click folder in file explorer and click 'Open in integrated terminal'
		- You can log in using CalNet ID, without using OTP.
	- Instructions:
		- Log in to Savio's Open Ondemand website:
			- https://ood.brc.berkeley.edu/
		- Click 'My Interactive Sessions' at top bar
		- In the 'Interactive Apps' column, click 'Code Server - exploration, debugging on shared nodes'
			- Make sure you click the right button; this application is free to use only when code-server running in *shared node*.  
		- Specify number of hours to use and click 'Launch'.
		- After 5 ~ 10 seconds, "Connect to VS Code" button will appear. click this button to use code-server web editor.
	
2. Terminal + SFTP
	- You can use terminal to ssh into Savio, and use SFTP programs to download/upload/edit files to Savio. 
	- Some popular SFTP programs:
		- For Mac: [Cyberduck](https://cyberduck.io/)(free), for Windows: [WinSCP](https://winscp.net/eng/index.php)(free)


NOTE: if you used Open Ondemand before and found that code-server was slow, I recommend checking it out again, as Savio has upgraded the nodes to run code-server faster.

For more information in using Open Ondemand code-server, please read [Savio Docs: Open OnDemand VSCode](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/ood/#code-server-vs-code).



## Storage options in Savio

### Which storage can we use?

In Savio, there are 2 main storage spaces that you can use.

1. Home directory
	- Path: `/global/home/users/jichan3751` (=`$HOME`)
	- Quota: 10GB
	- Data are backed up regularly.
	- Slow speed
2. Scratch directory
	- Path: `/global/scratch/users/$USER`
	- Quota: Unlimited 
		- WARNING: files can be removed if not accessed in 120 days.
			- (This policy is not effect)
	- Data are not backed up.
	- Fast speed

Since home directory quota is small, I recommend **using scratch directory for most of the cases** (for running experiments) and backing up the data regularly.

NOTE on storage speed: 
- Home and scratch directories are networked filesystems that can be accessed from any node(login nodes and compute nodes). 
Due to nature of accessing files through network, storage speed can slow down when large number of files are accessed (ex. loading conda environment, copying a large database folder).
	- It is likely that speed becomes faster if you access same files again, because 
	files are cached in the nodes when they first access.

- If you find storage is too slow for your application, there is also `/tmp/` directory that is not covered in this tutorial. This directory lies in SSDs physically attached to the compute node (not networked filesystem). It is fastest, but it cannot be accessed from other login / compute node. If you need this, take a look at [Savio: Staging Data for Computation](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/data/staging-data/)

For more information, take a look at the [Savio Docs](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/data/storing-data/).

## Initial setup 
This part explains recommended default initial setup.

### Install Miniconda

```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
Select default options when installing (by typing yes/y). 
Miniconda will be installed in `$HOME/miniconda3` directory.

You can access `conda` command after logging out and logging in again.

### Create symbolic link for scratch directory

```shell
$ ln -s /global/scratch/users/$USER ~/scratch
```
Now you can access your scratch directory by running `cd ~/scratch`.

### Moving conda, pip cache / other cache directories
When you run deep learning programs, software environments and packages from `conda` and `pip` can quickly consume lots of storage space in home directory.
To avoid this, we will move the packages cache / default environment location in home directory to scratch directory. Run each line of following to do this:

```shell
# Create directories for conda and pip
mkdir -p /global/scratch/users/$USER/conda_pkgs
mkdir -p /global/scratch/users/$USER/conda_envs
mkdir -p /global/scratch/users/$USER/pip

# Configure conda to use scratch space for pkgs and envs
conda config --add pkgs_dirs /global/scratch/users/$USER/conda_pkgs
conda config --add envs_dirs /global/scratch/users/$USER/conda_envs

# Configure pip to use scratch space for caching the software packages
echo "export PIP_CACHE_DIR=/global/scratch/users/$USER/pip" >> $HOME/.bashrc
```
Log out and log in again for settings to be applied.

NOTE : besides `conda` and `pip`, there could be other softwares that consume storage space in home directory. (ex. `Huggingface` and `PyTorch` datasets downloads large datasets and models used for training.) In such case, you can follow similar steps to move their cache directory to scratch, or specify the download directory to use scratch directory when you run them.

You can monitor your home directory usage by:
```
$ quota -s
Disk quotas for user <user> (uid <uid>): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
condo:/users/users
                  2034M      0K  30720M               0       0       0     
```

You can also check storage used by specific directory by `du -sh <dir>`:
```
$ du -sh ~/downloads
138M    /global/home/users/jichan3751/downloads
```

## Next steps

Go to the next part of the tutorial: [Train Resnet18 + CIFAR10 on Savio cluster](./train_cifar10.md).


