#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 12G
#SBATCH --partition shortgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --qos gpgpumse
#SBATCH -o /data/projects/punim1355/dalmiapriyam/slurmoutput/slurm-%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=dalmiap@student.unimelb.edu.au
module purge
module load gcc/8.3.0 fosscuda/2019b #torch/20200428
#module load tensorflow/2.3.1-python-3.7.4 #tesorflow-probability/0.9.0-python-3.7.4
module load pytorch/1.5.1-python-3.7.4
module load tensorflow-probability/0.9.0-python-3.7.4

#module load numpy/1.18.0-python-3.7.4
#module load protobuf/3.14.0
#module load protobuf-python/3.14.0

nvidia-smi > nvidia-smi.xt
python3 ./tests/tpred.py 10
