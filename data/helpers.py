import argparse
import logging

"""
Helper Functions and Variables for the trainers and agents.
""" 
parser = argparse.ArgumentParser(description="RL Agents",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id', type=str, help="%j to pass job id")
parser.add_argument('--size', type=int, default=5, help="size of the game map")
ARGS = parser.parse_args()

class dodict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
