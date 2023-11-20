#!/bin/bash
#SBATCH --nodes=1
#SBTACH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpgpudeeplearn
#SBATCH --partition=deeplearn
#SBATCH --time=20:00:00
#SBATCH -o ./slurmoutput/train-%j.out
echo 
echo -e "Name of the cluster on which the job is executing:\n\t $SLURM_CLUSTER_NAME"
echo -e "Number of CPUs on the allocated node: \n\t $SLURM_CPUS_ON_NODE"
echo -e "Number of CPUs requested per task: \n\t $SLURM_CPUS_PER_TASK"
echo -e "Numer of GPUs requested: \n\t $SLURM_GPUS"
echo -e "Requested GPU count per allocated node: \n\t $SLURM_GPUS_PER_NODE"
echo -e "Requested GPU count per allocated task:\n\t  $SLURM_GPUS_PER_TASK"
echo -e "The ID of the job allocation:\n\t  $SLURM_JOB_ID"
echo -e "Count of processors available to the job on this node:\n\t  $SLURM_JOB_CPUS_PER_NODE"
echo -e "Name of the job:\n\t  $SLURM_JOB_NAME"
echo -e "List of nodes allocated to the job:\n\t  $SLURM_JOB_NODELIST"
echo -e "Total number of nodes in the job’s resource allocation:\n\t  $SLURM_JOB_NUM_NODES"
echo -e "Name of the partition in which the job is running:\n\t  $SLURM_JOB_PARTITION"
echo -e "Minimum memory required per allocated CPU:\n\t  $SLURM_MEM_PER_CPU"
echo -e "Requested memory per allocated GPU:\n\t  $SLURM_MEM_PER_GPU"
echo -e "Total amount of memory per node that the job needs:\n\t  $SLURM_MEM_PER_NODE"
echo -e "List of nodes allocated to the job:\n\t  $SLURM_NODELIST"
echo -e "Total number of CPUs allocated:\n\t  $SLURM_NPROCS"
echo -e "Maximum number of MPI tasks (that’s processes): \n\t $SLURM_NTASKS"
echo -e "Number of tasks requested per core: \n\t $SLURM_NTASKS_PER_CORE"
echo -e "Number of tasks requested per GPU: \n\t $SLURM_NTASKS_PER_GPU"
echo -e "Number of tasks requested per node:\n\t  $SLURM_NTASKS_PER_NODE"
echo -e "The scheduling priority (nice value) at the time of job submission. This value is propagated to the spawned processes: \n\t $SLURM_PRIO_PROCESS"
echo -e "The MPI rank (or relative process ID) of the current process: \n\t $SLURM_PROCID"
echo -e "The directory from which SBATCH was invoked: \n\t $SLURM_SUBMIT_DIR"
echo -e "The Hostname of the computer from which SBATCH was invoked: \n\t $SLURM_SUBMIT_HOST"
echo -e "The process ID of the corresponding task: \n\t $SLURM_TASK_PID"

echo -e "\n\n\n LOADING MODULES: \n"

module purge 
module load GCCcore/11.3.0; module load Python/3.10.4
source ./.venv/bin/activate

echo -e "\n\n\n PYTHON SCRIPT OUTPUT: \n"

python3 ./train.py

my-job-stats -a -n -s