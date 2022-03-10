import os
import argparse 
import pdb
from game import Game

parser = argparse.ArgumentParser(description="Test game file")

parser.add_argument('size', type=int, help="Size(b) of the map: b * b")
parser.add_argument('-npred', default = 1, type=int, metavar="", help="Number of predators")
parser.add_argument('-nprey', default = 1, type=int, help="Number of preys")
args = parser.parse_args()


if __name__ == "__main__":
    breakpoint()
    game = Game(args.size, args.npred, args.nprey, 1)   
    game.render()    

    learning_rate = 0.01
    gamma = 0.9


