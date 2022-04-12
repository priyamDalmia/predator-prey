import os
import sys
sys.path.append(os.getcwd())

import logging 
import random
import time
import wandb
import numpy as np
from datetime import datetime

from game import Game
from data.replay_buffer import ReplayBuffer
from data.common import ARGS, dotdict
from agents.torch_dqn import Agent 
from agents.indp_dqn import DQNAgent
from agents.indp_ddqn import DDQNAgent
from agents.random_agent import RandomAgent 

def get_config():
    time = datetime.now().strftime("%d/%m %H:%M")
    decp = "torch:DQN"
    config = dict(
            # agent variables
            agenttype = "random",
            lr=0.0001,
            gamma=0.95,
            training=True,
            buffer_size = 10000,
            batch_size = 64,
            fc1_dims = 256,
            fc2_dims = 256,
            save_model = False,
            save_replay=True,
            # game variables
            size=ARGS.size,
            nprey=1,
            npred=1,
            nobstacles=0,
            nholes=0,
            winsize=5,
            # train and test variables
            epochs = 100,
            episodes = 500,
            train_steps = 10,
            # logging variables
            msg = f"torch DQN Agent, does not clear buffer",
            mode="online",
            decp = decp,
            run_name=f"{decp}:{time}:M{ARGS.size}",
            )
    return dotdict(config)

def get_logger(config):
    wandb.init(project="predator-prey", 
            notes=config.msg, 
            mode=config.mode,
            config=config)
    wandb.run.name = config.run_name
    return None

def initialize_agents(agent_ids, input_dims, output_dims, action_space, config):
    '''
    method to initialze the agent policies 
    '''
    agents = {}
    replay_mem = ReplayBuffer(config.buffer_size, config.batch_size, input_dims)
    for _id in agent_ids:
        if _id.startswith("predator"):
            obj = RandomAgent(input_dims, output_dims, action_space)
        else:
            if config.agenttype == "random":
                obj = RandomAgent(input_dims, output_dims, action_space)
            else:    
                obj = Agent(input_dims, output_dims, action_space,
                      False, memory=replay_mem, lr=config.lr)
        agents[_id] = obj
    return agents 

def run_episodes(env, agents, config):
    '''
    runs epsiodes with agents and return reward history
    '''
    steps_history = []
    rewards_history = []
    epsilon = 0
    for ep in range(config.episodes):
        observation = env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done:
            actions = {}
            # get actions for all agents 
            for _id in agents.keys():
                try:
                    actions[_id] = int(agents[_id].get_action(observation[_id]))
                except Exception as e:
                    print(e)
                    breakpoint()
            # step through the env
            rewards,  next_, done, info = env.step(actions)
            # store transition for each prey agent
            # update individual logs in here 
            # and add condition for not done
            for _id in agents.keys():
                if _id.startswith("prey") and config.agenttype != "random":
                    agents[_id].store_transition(observation[_id],
                            actions[_id],
                            rewards[_id],
                            next_[_id],
                            done)
                    total_reward += rewards[_id]
                    epsilon = agents[_id].epsilon
            # update observation 
            observation = next_
            steps+=1
        steps_history.append(steps)
        rewards_history.append(total_reward)
    # if save replays is true
    return steps_history, rewards_history, epsilon

def run_training(env, agents, config):
    loss_history = []
    for i in range(config.train_steps):
        for _id in agents.keys():
            if _id.startswith("prey"):
                loss = agents[_id].train_on_batch()
        loss_history.append(loss)
    return loss_history

def update_logs(epoch, steps, rewards, loss, epsilon):
    wandb.log(dict(epochs=epoch,
        steps=steps,
        rewards=rewards,
        loss=loss,
        epsilon=epsilon))

if __name__=="__main__":
    '''
    Testing and training a PREY policy on the predator prey env.
    Any modification for networked agents must be added here.
    '''
    # get config
    config = get_config()
    # init logger and wandb
    logger = get_logger(config)
    # init GAME and control variables
    env = Game(config)
    epochs = config.epochs
    # init agents 
    agent_ids = env.agent_ids
    action_space = env.action_space
    input_dims = env.observation_space.shape
    output_dims = len(action_space)
    agents = initialize_agents(agent_ids, input_dims, output_dims, 
            action_space, config)
    print("Training Agent")
    for epoch in range(epochs):
        # generate episodes 
        steps, rewards, epsilon = run_episodes(env, agents, config)
        # train on episodes
        loss = 0
        if config.training:
            loss = run_training(env, agents, config)
        loss_avg = np.mean(loss)

        steps_avg = np.mean(steps)
        rewards_avg = np.mean(rewards)
            
        # saves the last episode of the epoch
        if config.save_replay:
            info = dict(
                    steps_avg = float(steps_avg), 
                    rewards_avg= float(rewards_avg))
            filename = f"{config.decp}:{epoch}"
            env.record_episode(filename, info)
        # log the results
        print(f"epoch:{epoch}, average:{steps_avg}, loss:{loss_avg}")
        update_logs(epoch, steps_avg, rewards_avg, loss_avg, epsilon)
    wandb.finish()  