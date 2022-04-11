import os
import sys
import gym
import random
import logging 
import time
import numpy as np
import pdb
from datetime import datetime

sys.path.append(os.getcwd())
from helpers import dotdict
from data.replay_buffer import ReplayBuffer
from data.agent import BaseAgent
from agents.torch_dqn import Agent
from agents.random_agent import RandomAgent
from agents.tor_ac import ActorCriticAgent
from agents.tor_dqn import DQNAgent

import wandb
"""
Script for training agents on the OpenAI gym api.
## from gym import envs
## env.registry.all() -> list all environments.
"""


def get_config():
    time = datetime.now().strftime("%d/%m %H:%M")
    decp = "dqn:random"
    env = "CartPole-v1"
    config = dict(
            env=env,
            # agent variables
            agenttype = "dqn",
            lr=0.0001,
            gamma=0.95,
            training = True,
            buffer_size = 50000,
            batch_size = 64,
            fc1_dims = 256,
            fc2_dims = 256,
            save_model = False,
            save_replay = False,
            # train and test variables
            epochs = 1000,
            episodes = 1000,
            train_steps = 200,
            # logging variables
            wandb = True,
            msg = f"DQN Agent, Simple Network, Setting baselines.",
            mode="online",
            decp = decp,
            run_name=f"{env}:{decp}:{time}",
            loglevel=10, 
            logfile=f"logs/{decp}.log"
            )
    return dotdict(config)

def get_logger():
    if config.wandb:
        wandb.init(project="gym_benchmarks", 
                notes=config.msg, 
                mode=config.mode,
                config=config)
        wandb.run.name = config.run_name

    logger = logging.getLogger(__name__)
    logger.setLevel(config.loglevel)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(config.logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug(f"{__name__}:{config.msg}")
    logger.debug(f"Game|Size{config.size}|npred{config.npred}|nprey{config.nprey}")
    logger.debug(datetime.now().strftime("%d/%m %H:%M"))
    return logger


def shut_logger(config):
    if config.wandb:
        wandb.finish()
    logging.shutdown()

def initialize_agent(_id, input_dims, output_dims, action_space, config, logger):
    """
    method to initialize the agent policy 
    """
    replay_mem = ReplayBuffer(config.buffer_size, config.batch_size, input_dims)
    if config.agenttype == "random":
        agent = RandomAgent(_id, input_dims, output_dims, 
                action_space, False)
    else:
        agent = DQNAgent(_id, input_dims, output_dims, 
                action_space, False, memory=replay_mem)
    assert isinstance(agent, BaseAgent),"Error: Derive Agent from the BaseAgent class."
    logger.info(f"Agent Device: {agent.device}")
    return agent 

def run_episodes(env, agent, config):
    """
    runs epsiodes with agents and return reward history
    """
    steps_history = []
    rewards_history = []
    epsilon = 0
    for ep in range(config.episodes):
        observation = env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done:
            # get actions for all agents
            action = agent.get_action(observation)
            # step through the env
            next_, reward, done, info = env.step(action)
            # store transition
            agent.store_transition(observation,
                    action,
                    reward,
                    next_,
                    done)
            total_reward += reward
            epsilon = agent.epsilon
            # update observation 
            observation = next_
            steps+=1
        steps_history.append(steps)
        rewards_history.append(total_reward)
    # if save replays is true
    return steps_history, rewards_history, epsilon

def run_training(agent, config):
    loss_history = []
    for i in range(config.train_steps):
        loss = agent.train_on_batch()
        loss_history.append(loss)
    return loss_history

def update_logs(config, logger, epoch, steps, rewards, loss, epsilon):
    if config.wandb:
        wandb.log(dict(epochs=epoch,
            steps=steps,
            rewards=rewards,
            loss=loss,
            epsilon=epsilon))
    logger.info(f"EPOCHS: {epoch} \n ")
    logger.info(f"steps:{steps} rewards:{rewards} loss:{loss}")

if __name__=="__main__":
    # get config 
    config = get_config()
    # get logger    
    logger = get_logger()
    try:
        env = gym.make(config.env)
    except Exception as e:
        logger.error(e)
        logger.error("Gym Environment cannot be created!")
        sys.exit()
    # environment 
    input_dims = env.observation_space.shape
    output_dims = env.action_space.n
    action_space = [i for i in range(env.action_space.n)]
    # training variables 
    epochs = config.epochs
    episodes = config.episodes 
    train_steps = config.train_steps
    # initialize agent policy
    agent = initialize_agent("dqn_agent", input_dims, output_dims, 
            action_space, config, logger)
    print("Training Agent")
    for epoch in range(epochs):
        # generate episodes 
        steps, rewards, epsilon = run_episodes(env, agent, config)
        # train on episodes
        loss = 0
        if config.training:
            loss = run_training(agent, config)
        if (epoch%3) == 0:
            agent.update_epsilon()
        rewards_avg = np.mean(rewards)
        steps_avg =np.mean(steps)
        loss_avg = np.mean(loss)
        # saves the last episode of the epoch
        if config.save_replay:
            pass
            # gym save replay files 
        # log the results
        print(f"epoch:{epoch:.2f}, average:{steps_avg:.2f},rewards_avg:{rewards_avg:.2f}, loss:{loss_avg:.2f}, epsilon:{agent.epsilon:.2f}")
        update_logs(config, logger, epoch, steps_avg, rewards_avg, loss_avg, epsilon)
    shut_logger(config)
