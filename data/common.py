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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
