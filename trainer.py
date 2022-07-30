import os
import sys 

from data.game import Game

if __name__ == "__main__":
    print(" INSIDE TRAINER" )

    # Create the Game, Ethier Manually or using a
    # The Game can itself support multiple envs 
    env_name = "simple_pp"

    # Use a Static Method to list and see the tunable paramters 

    game = Game(env_name, config) q
