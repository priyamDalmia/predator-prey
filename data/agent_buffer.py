from dataclasses import dataclass
import os
import numpy as np

@dataclass
class Memory:
    state: obj
    action: obj
    reward: obj
    next_state: obj
    done: obj
    info: obj
    action_prob: obj


