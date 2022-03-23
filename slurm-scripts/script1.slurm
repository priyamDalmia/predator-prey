#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgputest
#SBATCH --gres=gpu:p100:1
#SBATCH --time 338:00:00
#SBATCH --cpus-per-task=1


module purge
module load fosscuda/2019b
module load tensorflow/2.1.0-python-3.7.4

python test.py 5 -agent-type "dqn" -loglevel 20 -m "DQN agent Random Actions SPARTAN"
