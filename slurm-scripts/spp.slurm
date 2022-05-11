#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --partition gpgpu
#SBATCH --qos gpgpuresplat
#SBATCH --gres=gpu:1
#SBATCH --account=punim1679
#SBATCH --time 20:00:00
#SBATCH --mem-per-cpu=6G

#SBATCH -o  /home/dalmiapriyam/bpp/slurmoutput/slurm-%j.out

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=dalmiap@student.unimelb.edu.au

module load pytorch/1.5.1-python-3.7.4

python3 experiments/1/main.py
#python3 trainers/train_agent.py
#python3 trainers/train_cent_pred.py
