import os
import sys
import gym
import numpy as np
sys.path.append(os.getcwd())
from data.helpers import dodict
from data.trainer import Trainer
from game.game import Game
from data.replay_buffer import ReplayBuffer
# Importing Agents
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_dqn import DQNAgent
from agents.tor_reinforce import RFAgent
from agents.tor_adv_ac import ACAgent

import pdb

# Training a Reinforce Agent.
# Default Agent Network
network_dims=dodict(dict(
        clayers=2,
        cl_dims=[3, 6, 12],
        nlayers=2,
        nl_dims=[256, 256]))
agent_network=dodict(dict(
        network_dims=network_dims))
config = dodict(dict(
        # Environment
        env="LunarLander-v2",
        # Training Control
        epochs=5000,
        episodes=1,
        train_steps=1,
        update_eps=1,
        training=True,
        save_replay=False,
        save_checkpoint=False,
        # Agent Control
        agent_type="REINFORCE",
        agent_class=ACAgent,
        load_model=False,
        lr=0.001, 
        gamma=0.99,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        agent_network=agent_network,
        buffer_size=1000,
        batch_size=64,
        # Log Control
        msg="message",
        notes="reinforce agent",
        project_name="gym-benchmarks",
        wandb=False,
        wandb_mode="offline",
        wandb_run_name="reinforce",
        log_level=10,
        log_file="logs/reinforce.log",
        ))

class train_gym(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_gym, self).__init__(config)
        self.config = config
        self.env = env
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        # initialize the agent
        self.agent = self.initialize_agents()
        self.checkpnt_state = self.agent.save_state()

    def train(self):
        rewards_hist = []
        steps_hist = []
        for epoch in range(self.config.epochs):
            loss = 0
            # Run Episodes
            steps, rewards, epsilon = self.run_episodes()
            # Train 
            if self.config.training:
                loss = self.run_training()
            if (epoch%self.config.update_eps) == 0:
                self.agent.update_eps()
            # Any Agent Specific Update goes here.
            rewards_hist.append(rewards)
            #reward_avg = np.mean(rewards)
            steps_hist.append(steps)
            #loss_avg = np.mean(loss)
            if self.config.save_replay:
                pass
            if self.config.save_checkpoint:
                pass
            if ((epoch+1)%100) == 0:
                info = dict(
                        steps = np.mean(steps_hist[-99:]),
                        rewards = np.mean(rewards_hist[-99:]))
                self.update_logs(epoch, info=info)
                print(f"Epochs:{epoch:4} | Steps:{info['steps']:4.2f} | Rewards:{info['rewards']:4.2f}")

    def initialize_agents(self):
        memory = ReplayBuffer(
                self.config.buffer_size,
                self.config.batch_size, 
                self.input_dims)
        if self.config.agent_type == "random":
            agent = RandomAgent(
                    str(self.config.agent_class),  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space)
            return agent
        agent = self.config.agent_class(
                str(self.config.agent_class), 
                self.input_dims,
                self.output_dims,
                self.action_space,
                memory=memory,
                **self.config)
        assert isinstance(agent, BaseAgent), "Agent not of class BaseAgent"  
        return agent

    def run_episodes(self):
        step_hist = []
        reward_hist = []
        epsilon = 0
        for ep in range(self.config.episodes):
            observation = self.env.reset()
            done = False
            steps = 0
            total_reward = 0
            while not done:
                action, probs = self.agent.get_action(observation)
                next_, reward, done, info = self.env.step(action)
                self.agent.store_transition(observation,
                        action,
                        reward,
                        next_,
                        done,
                        probs)
                total_reward += reward
                #epsilon = self.agent.epsilon
                observation = next_
                steps+=1
            step_hist.append(steps)
            reward_hist.append(total_reward)
        return step_hist, reward_hist, epsilon

    def run_training(self):
        loss_hist = []
        for i in range(self.config.train_steps):
            loss = self.agent.train_on_batch()
            loss_hist.append(loss)
        return loss_hist

    def save_replay(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

if __name__=="__main__":
    # Create the Environment object.
    try:
        env = gym.make(config.env)
    except Exception as e:
        print(e)
        print(f"Gym Environment:{config.env} could not be created!")
    input_dims = env.observation_space.shape
    output_dims = env.action_space.n
    action_space = [i for i in range(env.action_space.n)]
    trainer = train_gym(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()
