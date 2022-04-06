import os
import sys
import argparse

'''


ACTIONS : 0 = move_down
        : 1 = move_uo
        : 2 = move_right 
        : 3 = move_left
        : 4 = move_right
'''
ACTIONS = {
        0 : lambda x, s, p: (max(p, min(x[0]+1, s+p-1)), x[1]),
        1 : lambda x, s, p: (min(max(p, x[0]-1), s+p-1) ,x[1]),
        2 : lambda x, s, p: (x[0] ,max(p, min(x[1]+1, s+p-1))),
        3 : lambda x, s, p: (x[0] ,min(max(p, x[1]-1), s+p-1))
        }

ACTIONS_PREY = {
        '0' : lambda x, y, s, p: (max(p, min(x+1, s+p-1)), y),
        '1' : lambda x, y, s, p: (min(max(p, x-1), s+p-1) ,y),
        '2' : lambda x, y, s, p: (x ,max(p, min(y+1, s+p-1))),
        '3' : lambda x, y, s, p: (x ,min(max(p, y-1), s+p-1))
        }


# Setting up arguments to be parsed
parser = argparse.ArgumentParser(description="Test game file", 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('size', type=int, help="Size(b) of the map: b * b")
parser.add_argument('-npred', default = 1, type=int, metavar="", help="Number of predators")
parser.add_argument('-nprey', default = 1, type=int, metavar='', help="Number of preys")
parser.add_argument('-winsize', default=5, type=int, metavar="", help="Agent observation window size")
parser.add_argument('-agenttype', default="dqn", type=str, metavar="", help="Agent type (default: random)")
parser.add_argument('-logfile', default="logs/log.txt", type=str, metavar="", help="Log file name")
parser.add_argument('-loglevel', default=20, type=int, metavar="", help="Logging level of the program (_0s)")
parser.add_argument('-trainpred', default=False, type=bool, metavar="", help="Train predator agents (boolean)")
parser.add_argument('-trainprey', default=False, type=bool, metavar="", help="Train prey agents (boolean)")
parser.add_argument('-m', '--message', default="random", type=str, metavar="", help="Message for logger")
ARGS = parser.parse_args()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
