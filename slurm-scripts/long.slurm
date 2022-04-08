#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 12G
#SBATCH --partition longgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 30-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --qos gpgpumse
#SBATCH -o /data/projects/punim1355/dalmiapriyam/slurmoutput/slurm-%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=dalmiap@student.unimelb.edu.au
module purge
module load gcc/8.3.0 fosscuda/2019b
module load pytorch/1.5.1-python-3.7.4
module load tensorflow-probability/0.9.0-python-3.7.4

nvidia-smi >nvidia-smi.txt
python3 ./tests/tpred.py 10
