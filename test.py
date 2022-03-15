import os
import argparse 
import logging
import random 
import time
import pdb
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

# Initalizes agent polices 
def init_agents(agents_list):
    pass

if __name__ == "__main__":
    env = Game(args.size, args.npred, args.nprey, 1)   
      

    learning_rate = 0.01
    gamma = 0.9
    episodes = 1000
    
    agent_ids = env.agent_ids

    for i in range(episodes):

        step = 0
        done = env.reset()
        while not done:
            
            actions_t = []
            for agent in agent_ids:
                breakpoint()
                observation = env.observe(agent)
                actions_t.append(random.randint(0,3))
            
            env.render()
            time.sleep(0.3)
            rewards, done = env.step(actions_t)
            step+=1 


            
       

