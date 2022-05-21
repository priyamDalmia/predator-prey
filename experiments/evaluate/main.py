import os
import sys
sys.path.append(os.getcwd())
import yaml
import numpy as np
import pandas as pd
from evaluate import Evaluate
from data.helpers import dodict
from game.game import Game
from trainers.train_nac import train_agent
from agents.random_agent import RandomAgent
from agents.tor_naac import AACAgent
import argparse
import pdb
import logging 
from datetime import datetime

parser = argparse.ArgumentParser(description="experiments",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id', type=int, help="The configuration to run")
ARGS = parser.parse_args()

actor_network = dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))

config = dodict(dict(
        mode="eval",
        # Environment
        size=10,
        npred=1,
        nprey=1,
        winsize=9,
        nholes=0,
        nobstacles=0,
        map_="random",
        reward_mode="individual",
        advantage_mode=False,
        # Training control,
        epochs=2500,
        episodes=1,
        train_steps=1,
        update_eps=1,
        max_cycles=1000,
        training=True,
        train_type="predator",
        # Models
        log_level=10,
        print_console=True,
        ))

def get_logger(filename):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(10)
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(datetime.now().strftime("%d/%m %H:%M"))
    return logger

def shut_logger(logger):
    logger.handlers.clear()
    logging.shutdown()

if __name__=="__main__":
    config = config
    # Parse and Load Config File here.
    job_id = ARGS.id
    with open('experiments/evaluate/config.yaml') as f:
        job_data = yaml.load(f, Loader=yaml.FullLoader)
        config.update(job_data["experiments"]["game_config"])
    # Create and initialize Environments
    # Try passing Game Specific Config File - config.game
    # If Training; run trainers
    for i in range(config.eval_runs):
        config.update(job_data["experiments"][f"run_{i}"])
        try:
            env = Game(config)
        except:                
            print(f"Failed to initialzie Game Env.")
            sys.exit()
        logger = get_logger(config.eval_file)
        input_dims = env.observation_space.shape
        output_dims = len(env.action_space)
        action_space = env.action_space
        evaluate = Evaluate(config,
                env,
                input_dims=input_dims,
                output_dims=output_dims,
                action_space=action_space,
                logger = logger)
        logger.info(config.notes)
        logger.info(f"EVAL RUN: {i}")
        a = evaluate.evaluate()
        shut_logger(logger)
