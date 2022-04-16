#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 12G
#SBATCH --partition longgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 30-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --qos gpgpumse
#SBATCH -o  /home/dalmiapriyam/bpp/slurmoutput/slurm-%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=dalmiap@student.unimelb.edu.au

#module purge
#module load gcc/8.3.0 fosscuda/2019b 
#module load tensorflow/2.3.1-python-3.7.4 tensorflow-probability/0.9.0-python-3.7.4
#module load python/3.7.4
module load pytorch/1.5.1-python-3.7.4

python3 trainers/train_gym_v2.py
