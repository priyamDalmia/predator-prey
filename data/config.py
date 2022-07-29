from dataclasses import dataclass

from typing import List

@dataclass
class Config:
    env_name: str = "SimplePP"
    
    map_size: int = 6

    npred = 2
    nprey = 2 
    
    pred_vision = 3
    prey_vision = 3


