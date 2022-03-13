import os
import sys


'''
ACTIONS : 0 = move_up
        : 1 = move_down
        : 2 = move_right 
        : 3 = move_left
        : 4 = move_right
'''
ACTIONS_PRED = {
        '0' : lambda x, y, s: (max(0, min(x+1, s-1)), y),
        '1' : lambda x, y, s: (min(max(0, x-1), s-1) ,y),
        '2' : lambda x, y, s: (x ,max(0, min(y+1, s-1))),
        '3' : lambda x, y, s: (x ,min(max(0, y-1), s-1))
        }

ACTIONS_PREY = {
        '0' : lambda x, y, s: (max(0, min(x+1, s-1)), y),
        '1' : lambda x, y, s: (min(max(0, x-1), s-1) ,y),
        '2' : lambda x, y, s: (x ,max(0, min(y+1, s-1))),
        '3' : lambda x, y, s: (x ,min(max(0, y-1), s-1))
        }



