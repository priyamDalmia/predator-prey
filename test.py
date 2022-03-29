import os
import logging 
import numpy as np
import random
import time
from datetime import datetime

from game import Game
from data.replay_buffer import ReplayBuffer
from data.common import ARGS
from agents.indp_dqn import DQNAgent
from agents.indp_ddqn import DDQNAgent
from agents.random_agent import RandomAgent

# Setup logger
def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(ARGS.loglevel)
    formatter = logging.Formatter('%(message)s')
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
    training = {}
    input_dims = input_dims 
    output_dims = output_dims
    replay_mem = ReplayBuffer(buffer_size=10000, batch_size=64, state_size = input_dims)
    for _id in agent_ids: 
        if _id.startswith("predator"):
            training[_id] = ARGS.trainpred
            # if ARGS.agenttype == "random":
            obj = RandomAgent(input_dims, output_dims, action_space, False)
            # else:
                # obj = DQNAgent(input_dims, output_dims, action_space, False, memory=None)
            agents[_id] = obj
        else:
            training[_id] = ARGS.trainprey
            if ARGS.agenttype == "random":
                obj = RandomAgent(input_dims, output_dims, action_space, False)
            else:
                obj = DDQNAgent(input_dims, output_dims, action_space, memory=replay_mem) 
            agents[_id] = obj  
    
    return agents, training

if __name__=="__main__":
    '''
    A sample training loop for the API.

    1. Parallel run for the Predator-Prey Environment.
    2. All agent observations received in a dictionary.

    '''
    # Inintialize logger.
    logger = set_logger()
    logger.debug(f"{__name__}:{ARGS.message}")
    logger.debug(f"Game|Size{ARGS.size}|npred{ARGS.npred}|nprey{ARGS.nprey}")
    logger.debug(datetime.now().strftime("%d/%m %H:%M"))

    # Initialze game specific parameters here.
    action_space = [i for i in range(4)]
    average_steps = []

    # Initialize parameters for training here.
    max_cycles = 1000
    episodes = 10000

    # Create a environment object.
    env = Game(ARGS)
            
    # Initialize agent policies for training here.
    agent_ids = env.agent_ids
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    agents, training = initialize_agents(agent_ids, input_dims, output_dims, env.action_space, ARGS)
    tic  = time.perf_counter()
    
    for ep in range(episodes):
        # Reset environment and recieve initial observations.
        observation = env.reset()
        done = False
        steps = 0
        while not done:
            # clear the actions vector and iterate over agents to recieve the actions.
            # any messaging or reward sharing happens here.
            actions_i = {}
            for _id in agent_ids:
                try:
                    actions_i[_id] = int(agents[_id].get_action(observation[_id]))
                except Exception as e:
                    logger.error(e)
                    logger.error(f"Invalid action type for agent {_id}")
            # Step through the environment to receive the rewards, next_states, done, and info.
            rewards, next_obs, done, info = env.step(actions_i)
            # Store transition and train.
            for _id in agent_ids:
                agent = agents[_id]
                if _id.startswith("prey"):
                    try:
                        agent.store_transition(observation[_id], 
                            actions_i[_id], 
                            rewards[_id], 
                            next_obs[_id],
                            done)
                    except Exception as e:
                        print(e)
                        breakpoint()
            # Update observation
            observation = next_obs
           
            steps += 1
            #time.sleep(0.2)
            if done:
                #clear_lines(1)
         #       print(f"episode:{ep}, steps:{steps}")
                average_steps.append(steps)

        for _id in agent_ids:
            if training[_id]:
                agent = agents[_id]
                result = agent.train_on_batch()

        # Log results, save checkpoints and etc.    
        if (ep+1) % 5 == 0 :
            loss = result["loss"]
            logger.info(f"Episode: {ep}, loss:{loss:.3f}")
            logger.info(f"Episode: {ep+1} | Average steps: {np.average(average_steps[-50:])}")

    logger.info(f"Time Elapsed: {time.perf_counter()- tic}")

       

