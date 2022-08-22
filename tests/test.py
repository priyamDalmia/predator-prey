import os 
import sys
sys.path.append(os.getcwd())

from variations.simple_pp import SimplePP
from data.config import Config
from data.game import Game
from agents.random_agent import RandomAgent

from typing import List, Dict

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

def initialize_agents(agent_ids) -> Dict:
    all_agents = {}
    for current_id in agent_ids:
        input_space = game.observation_space(current_id)
        output_space = game.action_space(current_id)
        action_space = game.action_space(current_id)
        agent = RandomAgent(
                current_id, 
                input_space, 
                output_space, 
                action_space,
                memory = None)
        all_agents[current_id] = agent

    return all_agents

if __name__ == "__main__":
    breakpoint()
    # build game object here 
    config = Config()
    env = SimplePP(config.game_config)
    # wrap game 
    game = Game(config, env)

    # This is the part where you initialize the agents 
    # HERE
    all_agents = initialize_agents(game.agent_ids) 
    breakpoint()
    # This is the where we will train the agents 
    # HERE

    run_game(game)
    # create and visulize states 
