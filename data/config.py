from dataclasses import dataclass
from typing import List, Tuple, Callable

@dataclass
class TrainerConfig:
    trainer: bool = False

@dataclass
class GameConfig:
    env_name: str = "simplePP"
    verbose: bool = False

    npred: int = 10
    nprey: int = 10    
    pred_vision: int = 3
    prey_vision: int = 3

    time_mode: Tuple[bool, float] = (True, 500)
    action_mode: int = None 
    reward_mode: int = None

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


