import os 
import sys
sys.path.append(os.getcwd())

from variations.simple_pp import SimplePP
from data.config import Config
from data.game import Game
from data.helpers import *
from agents.random_agent import RandomAgent
from agents.tor_ppo import PPOAgent

from typing import List, Dict

network_dims=dodict(dict(
    clayers=2,
    cl_dims=[6, 12],
    nlayers=2,
    nl_dims=[256, 256]))

agent_network = dodict(dict(
    network_dims=network_dims))

def run_game(game):

    breakpoint()
    done = game.is_terminal()
    observations = game.reset()
    agent_ids = game.agent_ids
    while not game.is_terminal():
        game.render()
        actions = {}
        for agent_id in agent_ids:
            actions[agent_id] =\
                    game.action_space(agent_ids[0]).sample()
        game.step(actions) 
        obs = game.get_observations()
        breakpoint() 

def initialize_agents(agent_ids, agent_type="random") -> Dict:
    all_agents = {}
    for current_id in agent_ids:
        input_space = game.observation_space(current_id)
        output_space = game.action_space(current_id)
        action_space = game.action_space(current_id)
        
        if agent_type == "PPO":
            agent = PPOAgent(
                    current_id,
                    input_space, 
                    output_space,
                    agent_network = agent_network) 
        else:
            agent = RandomAgent(
                    current_id, 
                    input_space, 
                    output_space, 
                    action_space,
                    memory = None)
        all_agents[current_id] = agent

    return all_agents

def run_episodes(num_episodes, game, all_agents):
    breakpoint()
    for episode in range(num_episodes):
        game.reset()
        done = game.is_terminal()
        # THIS IS THE EXCEUTION OF A SINGLE EPISODE!
        while not done:
            current_observations = game.get_observations()
            current_action = {}

            for agent_id in game.agent_ids:
                current_action[agent_id] =\
                        all_agents[agent_id].get_action(current_observations[agent_id])
            breakpoint()
            game.step(current_action)

            # THIS IS WHERE YOU STORE TRANSITIONS 
            rewards = game.last_rewards()
            next_observations = game.get_observations()

            for agent_id in game.agent_ids:
                # transition tuple 
                agent = all_agents[agent_id]
                state = current_observations[agent_id]
                action = current_action[agent_id]
                reward = rewards[agent_id]
                next_state = next_observations[agent_id]
                is_alive = game.is_alive(agent_id)
                agent.store_transition((state, action, reward, (1-is_alive), next_state))

            game.render()

            done = game.is_terminal()
    pass

def run_training(all_agents):
    # TRAIN ALL AGENTS 
    for agent in all_agents:
        agent.train_step()

if __name__ == "__main__":
    breakpoint()
    # build game object here 
    config = Config()
    env = SimplePP(config.game_config)
    # wrap game 
    game = Game(config, env)

    # This is the part where you initialize the agents 
    # HERE
    all_agents = initialize_agents(game.agent_ids, agent_type="PPO") 
    # This is the where we will train the agents 
    # HERE
    epochs = 10
    num_episodes = 1
    # training happens in two steps 
    for i in range(epochs):
        run_episodes(num_episodes, game, all_agents)
        run_training(all_agents)
    # run_game(game)
    # create and visulize states 
