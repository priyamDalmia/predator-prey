# tests the compatitbilty of the envs with algorithms 
# 1. Independent Q-learning
# 2. Parameter Sharing, with Q-learning and Actor Critic
# 3. Centralized Actor Critic 

from pettingzoo.test import parallel_api_test
from environments.discrete_pp_v1 import discrete_pp_v1

def test_env_compatibilty():
    config = dict(
        map_size = 20,
        reward_type = "type_1",
        max_cycles = 10000,
        npred = 2,
        pred_vision = 2,
        nprey = 6,
        prey_type = "static",
        render_mode = None
    )
    env = discrete_pp_v1(config)