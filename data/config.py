from dataclasses import dataclass
from typing import List, Tuple, Callable
from data.utils import *

@dataclass
class TrainerConfig:
    trainer: bool = False

@dataclass
class GameConfig:
    env_name: str = "simplePP"
    verbose: bool = False
    render: bool = True
    render_info: bool = True
    
    map_size: int = 10
    max_steps: int = 500

    npred: int = 3
    nprey: int = 3   
    pred_vision: int = 5
    prey_vision: int = 5

    time_mode: Tuple[bool, float] = (True, 500)
    action_mode: int = action_group_random
    reward_mode: int = reward_individual
    health_mode: int = health_standard

@dataclass
class AgentConfig:
    pass
    
@dataclass
class ReplayBufferConfig:
    pass

@dataclass
class GPUConfig:
    pass

@dataclass
class Config:
    game_config: GameConfig = GameConfig()
    trainer_config: TrainerConfig = TrainerConfig()
    buffer_config: ReplayBufferConfig = ReplayBufferConfig()
    gpu_config: GPUConfig = GPUConfig()
    agent_config: AgentConfig = AgentConfig()
