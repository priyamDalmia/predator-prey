import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from evaluate import Evaluate
from data.helpers import dodict
from game.game import Game
from trainers.train_agent import train_agent
from agents.random_agent import RandomAgent
from agents.tor_adv_ac import AACAgent

import pdb

actor_network = dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))

config = dodict(dict(
        mode="train",
        # Environment
        size=15,
        npred=2,
        nprey=5,
        winsize=9,
        nholes=0,
        nobstacles=0,
        map_="random",
        # Training control,
        epochs=2500,
        episodes=1,
        train_steps=1,
        update_eps=1,
        max_cycles=1000,
        training=True,
        train_type="predator",
        eval_pred=False,
        eval_prey=False,
        # Agent Control
        class_pred=AACAgent,
        class_prey=RandomAgent,
        agent_type="adv-ac",
        agent_network=actor_network,
        lr=0.0005, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        batch_size=64,
        buffer_size=1500,
        # Models
        replay_dir="experiments/2/results/",
        checkpoint_dir="experiments/2/policies/",
        load_prey=False, 
        load_pred=False,# "experiments/1/policies/predator_0-10-1ac-1rand-2399-17",
        # Log Control
        _name="15-2ac-5rand",
        save_replay=True,
        save_model=True,
        log_freq=200,
        wandb=True,
        wandb_mode="online",
        entity="rl-multi-predprey",
        project_name="experiment 2",
        notes="2AAC vs 5RAND Pred Test - Team Rewards",
        log_level=10,
        log_file="logs/exp_2.log",
        print_console=True,
        ))

if __name__=="__main__":
    config = config
    # Parse and Load Config File here.

    # Create and initialize Environments
    # Try passing Game Specific Config File - config.game
    try:
        env = Game(config)
    except:
        print(f"Failed to initialzie Game Env.")
        sys.exit()
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space

    # If Training; run trainers
    if config.mode == "train":
        trainer = train_agent(config,
                env,
                input_dims=input_dims,
                output_dims=output_dims,
                action_space=action_space)
        trainer.train()
        trainer.shut_logger()
    # Else Evalaute; run evaulate
    else:
        evaluate = Evaluate(config,
                env,
                input_dims=input_dims,
                output_dims=output_dims,
                action_space=action_space)
        evaluate.evaluate()
        pass
