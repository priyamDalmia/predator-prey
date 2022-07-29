import os 
from variations.simplePP import SimplePP
from data.config import Config
from data.game import Game

def run_game(game):

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
        breakpoint() 


if __name__ == "__main__":

    # build game object here 
    config = Config()

    env = SimplePP(config)
    # wrap game 
    game = Game(config, env)
    run_game(game)
    # create and visulize states 
    
