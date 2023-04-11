import os 
import sys
sys.path.append(os.getcwd())

import wandb
from variations.simple_pp import SimplePP
from data.config import Config
from data.game import Game
from data.helpers import *
from agents.random_agent import RandomAgent
from agents.tor_a3c import A3CAgent
from statistics import mean
from typing import List, Dict

# network_dims=dodict(dict(
#     clayers=2,
#     cl_dims=[6, 12],
#     nlayers=2,
#     nl_dims=[256, 256]))

# agent_network = dodict(dict(
#     network_dims=network_dims))


class Trainer:
    def __init__(self, config):
        # build game object here 
        self.config = config

        # wandb
        if self.config.trainer_config.wandb:
            wandb.init(
                project = self.config.trainer_config.project_name,
                config = self.config.trainer_config
            )
    
    def train(self):
        # wrap game 
        env = SimplePP(self.config.game_config)
        game = Game(self.config, env)

        # This is the part where you initialize the agents 
        # HERE
        all_agents = self.initialize_agents(game, game.agent_ids, agent_type="A3C") 
        # This is the where we will train the agents 
        # HERE
        epochs = self.config.trainer_config.epochs
        num_episodes = self.config.trainer_config.episodes
        # training happens in two steps 
        for i in range(epochs):
            steps = self.run_episodes(num_episodes, game, all_agents)
            results = self.run_training(all_agents)
            results['steps'] = steps
            results['epoch'] = i

            if self.config.trainer_config.wandb:
                wandb.log(results)
                print(f"Epoch {results['epoch']} : {results['steps']}")
            else:
                print(f"Epoch {results['epoch']} : {results['steps']}")

        pass

    def initialize_agents(self, game, agent_ids, agent_type="random") -> Dict:
        all_agents = {}
        for current_id in agent_ids:
            input_space = game.observation_space(current_id)
            output_space = game.action_space(current_id)
            action_space = game.action_space(current_id)
            
            if "predator" in current_id:
                agent = A3CAgent(
                        current_id, 
                        input_space, 
                        output_space, 
                        action_space,
                        memory = "self")
            else:
                agent = RandomAgent(
                        current_id, 
                        input_space, 
                        output_space, 
                        action_space,
                        memory = None)
            all_agents[current_id] = agent

        return all_agents

    def run_episodes(self, num_episodes, game, all_agents):
        steps = []
        for episode in range(num_episodes):
            step = 0
            current_observations, dones = game.reset()
            done = game.is_terminal()
            # THIS IS THE EXCEUTION OF A SINGLE EPISODE!
            while not done:
                step += 1
                if game.game_config.render:
                    game.render()
                current_observations = game.get_observations()
                current_action = {}

                for agent_id in game.agent_ids:
                    current_action[agent_id] =\
                            all_agents[agent_id].get_action(current_observations[agent_id])
                next_observations, rewards, dones, info = game.step(current_action)

                # THIS IS WHERE YOU STORE TRANSITIONS 
                # rewards_1 = game.last_rewards()
                # next_observations_1 = game.get_observations()
                done = game.is_terminal()

                for agent_id, agent in all_agents.items():
                    if not agent.trains and dones[agent_id]:
                        continue
                    # transition tuple 
                    state = current_observations[agent_id]
                    action = current_action[agent_id]
                    reward = rewards[agent_id]
                    next_state = next_observations[agent_id]
                    is_alive = game.is_alive(agent_id)
                    agent.store_transition((state, action, reward, (1-is_alive), next_state))
            steps.append(step)
        return mean(steps)

    def run_training(self, all_agents):
        # TRAIN ALL AGENTS 
        train_info = dict()
        for agent_id, agent in all_agents.items():
            if agent.trains:
                info = agent.train_step()
                train_info[agent_id] = info
        return train_info

if __name__ == "__main__":
    trainer = Trainer(Config())
    trainer.train()
