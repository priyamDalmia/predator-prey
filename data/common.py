import os
import sys
import argparse

'''


ACTIONS : 0 = move_up
        : 1 = move_down
        : 2 = move_right 
        : 3 = move_left
        : 4 = move_right
'''
ACTIONS = {
        '0' : lambda x, y, s, p: (max(p, min(x+1, s+p-1)), y),
        '1' : lambda x, y, s, p: (min(max(p, x-1), s+p-1) ,y),
        '2' : lambda x, y, s, p: (x ,max(p, min(y+1, s+p-1))),
        '3' : lambda x, y, s, p: (x ,min(max(p, y-1), s+p-1))
        }

ACTIONS_PREY = {
        '0' : lambda x, y, s, p: (max(p, min(x+1, s+p-1)), y),
        '1' : lambda x, y, s, p: (min(max(p, x-1), s+p-1) ,y),
        '2' : lambda x, y, s, p: (x ,max(p, min(y+1, s+p-1))),
        '3' : lambda x, y, s, p: (x ,min(max(p, y-1), s+p-1))
        }


# Setting up arguments to be parsed
parser = argparse.ArgumentParser(description="Test game file")
parser.add_argument('size', type=int, help="Size(b) of the map: b * b")
parser.add_argument('-npred', default = 1, type=int, metavar="", help="Number of predators")
parser.add_argument('-nprey', default = 1, type=int, metavar='', help="Number of preys")
parser.add_argument('-win_size', default=3, type=int, metavar="", help="Agent observation window size")
ARGS = parser.parse_args()


