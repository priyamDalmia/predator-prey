import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
from data.helpers import dodict
from data.trainer import Trainer
from game.game import Game
from data.replay_buffer import ReplayBuffer
# Importing Agents
from data.agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.tor_dqn import DQNAgent
import pdb

agent_network = {}
config = dodict(dict(
        # Environment
        size=10,
        npred=1,
        nprey=1,
        winsize=5,
        nholes=0,
        nobstacles=0,
        env="CartPole-v1",
        # Training Control
        epochs=1,
        episodes=1,
        train_steps=1,
        update_eps=1,
        training=False,
        save_replay=False,
        save_checkpoint=False,
        # Agent Control
        agent_type="random",
        pred_class=RandomAgent,
        prey_class=RandomAgent,
        lr=0.0001, 
        gamma=0.95,
        epislon=0.95,
        epsilon_dec=0.99,
        epsilon_update=10,
        agent_network=agent_network,
        buffer_size=100000,
        batch_size=64,
        # Log Control
        msg="message",
        notes="random",
        project_name="gym-benchmarks",
        wandb=False,
        wandb_mode="offline",
        wandb_run_name="random",
        log_level=10,
        log_file="logs/random.log",
        ))

class train_prey(Trainer):
    def __init__(self, config, env, **env_specs):
        super(train_prey, self).__init__(config)
        self.config = config
        self.env = env
        self.input_dims = env_specs["input_dims"]
        self.output_dims = env_specs["output_dims"]
        self.action_space = env_specs["action_space"]
        self.logger = self.get_logger()
        # initialize the agent
        self.agent_ids = env.agent_ids
        self.agents = self.initialize_agents()
# Fix 
        self.checkpnt_state = None

    def train(self):
        rewards_hist = []
        for epoch in range(self.config.epochs):
            loss = 0
            # Run Episodes
            steps, rewards, epsilon = self.run_episodes()
            # Train 
            if self.config.training:
                loss = self.run_training()
            if (epoch%self.config.update_eps):
                for _id in self.agents:
                    self.agent[_id].update_epsilon()
            # Any Agent Specific Update goes here.
            breakpoint()
            reward_avg = np.mean(rewards)
            step_avg = np.mean(steps)
            loss_avg = np.mean(loss)
            rewards_hist.append(reward_avg)
            if self.config.save_replay:
                pass
            if self.config.save_checkpoint:
                pass

    def initialize_agents(self):
        agents = {}
        memory = ReplayBuffer(
                self.config.buffer_size,
                self.config.batch_size, 
                self.input_dims)
        breakpoint()
        for _id in self.agent_ids:
            if _id.startswith("predator"):
                agent = self.config.pred_class(
                    _id,  
                    self.input_dims,
                    self.output_dims, 
                    self.action_space,
                    memory = memory,
                    config = self.config)
            else:
                agent = self.config.prey_class(
                    _id, 
                    self.input_dims,
                    self.output_dims,
                    self.action_space,
                    memory = memory,
                    config = self.config)
                self.log("Agent {_id}, Device {agent.device}")
            assert isinstance(agent, BaseAgent), "Error: Derive agent from BaseAgent!"
            agents[_id] = agent
        return agents

    def run_episodes(self):
        step_hist = []
        reward_hist = []
        epsilon = 0
        for ep in range(self.config.episodes):
            observation = self.env.reset()
            done = False
            steps = 0
            all_rewards = []
            while not done:
                actions = {}
                # Get actions for all agents.
                for _id in self.agents:
                    actions[_id] = self.agents[_id].get_action(observation[_id])
                breakpoint()
                rewards, next_, done, info = self.env.step(actions)
                for _id in self.agents:
                    self.agents[_id].store_transition(observation[_id],
                        actions[_id],
                        rewards[_id],
                        next_[_id],
                        done)
                all_rewards.append(rewards)
                observation = next_
                steps+=1
            epsilon = self.agents["prey_0"].epsilon
            step_hist.append(steps)
            reward_hist.append(
                    pd.DataFrame(rewards[0]).sum(axis=0).to_dict())
        return step_hist, reward_hist, epsilon
    
    # Modify for multiple agents 
    def run_training(self):
        loss_hist = []
        for i in range(self.config.train_steps):
            for _id in self.agents:
                if _id.startswith("prey"):
                    loss = self.agents[_id].train_on_batch()
            loss_hist.append(loss)
        return loss_hist

    def save_replay(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

if __name__=="__main__":
    breakpoint()
    # Create the Environment object.
    try:
        env = Game(config)
    except:
        print(f"Gym Environment:{config.env} could not be created!")
    input_dims = env.observation_space.shape
    output_dims = len(env.action_space)
    action_space = env.action_space
    trainer = train_prey(config, 
            env, 
            input_dims=input_dims, 
            output_dims=output_dims,
            action_space = action_space)
    trainer.train()
    trainer.shut_logger()
