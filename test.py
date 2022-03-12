import os
import argparse 
import pdb
import logging
import random 

from game import Game
from agents import indp_dqn

# Setting up arguments to be parsed
parser = argparse.ArgumentParser(description="Test game file")
parser.add_argument('size', type=int, help="Size(b) of the map: b * b")
parser.add_argument('-npred', default = 1, type=int, metavar="", help="Number of predators")
parser.add_argument('-nprey', default = 1, type=int, metavar='', help="Number of preys")
args = parser.parse_args()

# Setting up logger
logging.basicConfig(level=logging.DEBUG, filename="./logs/tests.log")

if __name__ == "__main__":
    env = Game(args.size, args.npred, args.nprey, 1)   
      

    learning_rate = 0.01
    gamma = 0.9
    episodes = 1000
    
    agents_list = env.agents_list
    breakpoint()

    for i in range(episodes):
        
        done = env.reset()
        while not done:
            
            actions_t = []
            for agent in agents_list:
                actions_t.append(random.randint(0,3))
            
            breakpoint()
            rewards, done = env.step(actions_t)
            
            
       

