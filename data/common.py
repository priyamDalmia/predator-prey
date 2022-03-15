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



