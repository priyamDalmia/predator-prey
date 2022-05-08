import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pdb

from evaluate import Evaluate
from trainers.train_agent import train_agent

if __name__=="__main__":
    breakpoint()
    config = None
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
