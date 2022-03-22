import os
import logging 
import numpy as np
import random
import time
from datetime import datetime

from game import Game
from data.common import ARGS
from agents.indp_dqn import Agent
from agents.random_agent import RandomAgent

# Setup logger
def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(ARGS.loglevel)
    formatter = logging.Formatter('%(name)s:%(message)s')
    file_handler = logging.FileHandler(ARGS.logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Clear lines.
def clear_lines(n_lines=6):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2k'
    for i in range(n_lines):
        print(LINE_UP, end=LINE_CLEAR)

def initialize_agents(agent_ids: list, input_dims, output_dims, action_space, ARGS) -> dict:
    '''
    Function to initialize the agent policies, and return a 
    dict of policies.
    '''
    agents = {}
    input_dims = input_dims 
    output_dims = output_dims

    for _id in agent_ids: 
        if _id.startswith("predator"):
            if ARGS.agent_type == "random":
                obj = RandomAgent(input_dims, output_dims, action_space, False)
            else:
                obj = Agent(input_dims, output_dims, action_space, False)
            agents[_id] = obj
        else:
            if ARGS.agent_type == "random":
                obj = RandomAgent(input_dims, output_dims, action_space, False)
            else:
                obj = Agent(input_dims, output_dims, action_space, False) 
            agents[_id] = obj  
    
    return agents

if __name__=="__main__":
    '''
    A sample training loop for the API.

    1. Parallel run for the Predator-Prey Environment.
    2. All agent observations received in a dictionary.

    '''
    # Inintialize logger.
    logger = set_logger()
    logger.info(f"{__name__}:{ARGS.message}")
    logger.info(f"Game|Size{ARGS.size}|npred{ARGS.npred}|nprey{ARGS.nprey}")
    logger.info(datetime.now().strftime("%d/%m %H:%M"))

    # Initialze game specific parameters here.
    action_space = [i for i in range(4)]
    average_steps = []

    # Initialize parameters for training here.
    max_cycles = 10000
    episodes = 10000
    training = False

    # Create a environment object.
    env = Game(ARGS)
            
    # Initialize agent policies for training here.
    agent_ids = env.agent_ids
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    agents = initialize_agents(agent_ids, input_dims, output_dims, env.action_space, ARGS)
    
    print("Training_loop") 
    for ep in range(episodes):
        # Reset environment and recieve initial observations.
        observation = env.reset()
        done = False
        steps = 0
        tic  = time.perf_counter()
        while not done:
            # clear the actions vector and iterate over agents to recieve the actions.
            # any messaging or reward sharing happens here.
            actions_i = {}
            for _id in agent_ids:
                #agents[_id].get_action(observation[_id]) 
                #actions_i.append(agents[_id].get_action(observation[_id]))
                actions_i[_id] = random.randint(0, 3)
            # Step through the environment to receive the rewards, next_states, done, and info.
            rewards, next_obs, done, info = env.step(actions_i)
            
            # Store transition
            # Update observation

            # Perform Replay and training step here
            if training:
                pass
            
            steps += 1
            #time.sleep(0.2)
            if done:
                clear_lines(1)
                print(f"episode:{ep}, steps:{steps}")
                average_steps.append(steps)
            
        # Log results, save checkpoints and etc.    
        if (ep+1) % 100 == 0 :
            logger.info(f"Episode: {ep+1} | Average steps: {np.average(average_steps[-50:])}")

            
       

