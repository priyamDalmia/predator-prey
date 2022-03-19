import os
import logging 
import numpy as np
import random
import time

from game import Game
from data.common import ARGS
from agents.indp_dqn import Agent

# A method to clean a number of lines.
def clear_lines(n_lines=6):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2k'
    for i in range(n_lines):
        print(LINE_UP, end=LINE_CLEAR)

def initialize_agents(agent_ids: list, input_dims, output_dims) -> dict:
    '''
    Function to initialize the agent policies, and return a 
    dict of policies.
    '''
    input_dims = input_dims 
    output_dims = output_dims
    for i in agent_ids: 
        pass

if __name__=="__main__":
    '''
    A sample training loop for the API.

    1. Parallel run for the Predator-Prey Environment.
    2. All agent observations received in a dictionary.

    '''
    # Initialze game specific parameters here.
    action_space = [i for i in range(4)]
    
    # Initialize parameters for training here.
    max_cycles = 10000
    episodes = 100

    # Create a environment object.
    env = Game(ARGS)
            
    # Initialize agent policies for training here.
    agent_ids = env.agent_ids
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    agents = initialize_agents(agent_ids, input_dims, output_dims)
    
    
    for ep in range(episodes):
        # Reset environment and recieve initial observations.
        observation = env.reset()
        done = False
        steps = 0
        tic  = time.perf_counter()
        for i in range(max_cycles):
            env.render()
            # clear the actions vector and iterate over agents to recieve the actions.
            # any messaging or reward sharing happens here.
            actions_i = {}
            for _id in agent_ids:
            
                #actions_i.append(agents[_id].get_action(observation[_id]))
                actions_i[_id] = random.randint(0, 3)
            # Step through the environment to receive the rewards, next_states, done, and info.
            rewards, next_states, done, info = env.step(actions_i)
            steps += 1
            time.sleep(0.2)
            if done:
                self.render()
                print("All deers captured")
                print(f"Game finished in {steps} steps. Time take: {time.perf_counter() - tic}")
                input("Press Enter for next episode...")
                break
            clear_lines(10)
            

