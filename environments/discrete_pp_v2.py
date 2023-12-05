from environments.discrete_pp_v1 import discrete_pp_v1
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import os 
import time 

class discrete_pp_v2(discrete_pp_v1):
    """Discerete Space (2D) Predator Prey Environment
    Predators and Preys are randomly placed on a 2D grid.
    Predators must capture Prey.
    Predators are the agents.
    Preys are either fixed or move randomly.
    Game ends when all Preys are captured or max_cycles is reached.
    """

    # from data.utils import Predator, Prey
    NUM_CHANNELS = 3
    GROUND_CHANNEL = 0
    PREDATOR_CHANNEL = 1
    PREY_CHANNEL = 2

    # GRID (0, 0) : UPPER LEFT, (N , N) : LOWER RIGHT.
    NUM_ACTIONS = 5
    ACTIONS = {
            0: lambda pos_x, pos_y: (pos_x, pos_y), # STAY
            1: lambda pos_x, pos_y: (pos_x - 1, pos_y), # LEFT
            2: lambda pos_x, pos_y: (pos_x + 1, pos_y), # RIGHT
            3: lambda pos_x, pos_y: (pos_x, pos_y + 1), # DOWN
            4: lambda pos_x, pos_y: (pos_x, pos_y - 1), # UP
            }

    STR_TO_ACTION = {
        "STAY": 0,
        "LEFT": 1,
        "RIGHT": 2,
        "DOWN": 3,
        "UP": 4,
    }

    ACTION_TO_STR = {
        0: "STAY",
        1: "LEFT",
        2: "RIGHT",
        3: "DOWN",
        4: "UP",
    }
    
    kill_area_slice = lambda state, position, channel: state[
            position[0] - 2: position[0] + 3,
            position[1] - 2: position[1] + 3,
            channel].copy()
    
    def __init__(self, *args, **kwargs)-> None:
        super().__init__()
        self._map_size = kwargs.get("map_size", 10)
        self._max_cycles = kwargs.get("max_cycles", 100)
        self._npred = kwargs.get("npred", 2)
        self._pred_vision = kwargs.get("pred_vision", 2)
        self._nprey = kwargs.get("nprey", 6)
        self._prey_type = kwargs.get("prey_type", 'static') # random or fixed
        self._reward_type = kwargs.get("reward_type", "type_1") # type_1 or type_2
        # init the reward function here; reward function will distribute the reward only. 
        if self._reward_type == "type_1":
            self._reward_func = self.reward_dist_1
        elif self._reward_type == "type_2":
            self._reward_func = self.reward_dist_2
        elif self._reward_type == "type_3":
            self._reward_func = self.reward_dist_3
        else:
            raise ValueError(f"Reward type {self._reward_type} not supported.")
        # build base game state here 
        self._possible_agents = list(set([f"predator_{i}" for i in range(self._npred)]))
        self._map_pad = self._pred_vision
        arr = np.zeros(
            (self._map_size, self._map_size),
            dtype = np.int32)
        # define what the observation looks like 
        self._base_state = np.dstack([
            # a channel with 1s padded = ground channel
            np.pad(arr, self._map_pad, 'constant', constant_values=1),
            # a channel with 0s padded = unit (predator channel)
            np.pad(arr, self._map_pad, 'constant', constant_values=0),
            # a channel with 0s padded = unit (prey channel)
            np.pad(arr, self._map_pad, 'constant', constant_values=0),
        ])
        self._state_space = self._base_state.shape
        self._state_space_center = self._map_pad + self._map_size //  2
        
        window_l = 2 * self._pred_vision + 1 
        self._observation_space = Box(
            low=0,
            high=1,
            shape=(window_l * window_l * self.NUM_CHANNELS,),
            dtype=np.int32,
        )
        self._action_space = Discrete(self.NUM_ACTIONS)
        self._observation_spaces = {
            agent: self._observation_space for agent in self._possible_agents
        }
        self._action_spaces = {
            agent: self._action_space for agent in self._possible_agents
            }
        
        self.take_action = lambda position, action: self.ACTIONS[action](*position)
        self._metadata = {
            "name": "discrete_pp_v0",
            "render.modes": ["human", "rgb_array"],
            "map_size": self._map_size,
            "reward_type": self._reward_type,
            "max_cycles": self._max_cycles,
            "npred": self._npred,
            "nprey": self._nprey,
            "pred_vision": self._pred_vision,
            "prey_type": self._prey_type,}
        
        if self._npred > 2:
            raise NotImplementedError("Only 2 predators supported for now. Redo Reward func2 for compatibility")

       
    def get_observation(self, agent_id):
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} is not alive or does not exist.")

        agent = self._predators[agent_id]
        pos_x, pos_y = agent.position
        window_l = 2 * self._pred_vision + 1
        observation = self._global_state[
            pos_x - self._pred_vision: pos_x + self._pred_vision + 1,
            pos_y - self._pred_vision: pos_y + self._pred_vision + 1,
            :].copy()
        return observation.reshape(-1)

if __name__ == "__main__":
    config = dict(
        map_size = 15,
        reward_type = "type_2",
        max_cycles = 1000,
        npred = 2,
        pred_vision = 4,
        nprey = 6,
        prey_type = "static",
        render_mode = None
    )
    from environments.discrete_pp_v1 import FixedSwingAgent, FollowerAgent, ChaserAgent
    env = discrete_pp_v2(**config)
    fixed_agent = FixedSwingAgent(env)
    follower_agent = FollowerAgent(env)
    chaser_agent = ChaserAgent(env) 
    print(f"Env name: {env.__name__()}, created!")
    
    # test action and observation spaces 
    assert isinstance(env.observation_spaces, dict)
    assert isinstance(env.action_spaces, dict)

    # if map size is 20 and vision size is 2,
    # then the observation space should be 5x5x3
    window_l = 2 * env._pred_vision + 1
    assert all([env.action_spaces[agent_id].n == env.NUM_ACTIONS \
                for agent_id in env._possible_agents])

    reward_hist = []
    steps_hist = []
    assists_hist = []
    kills_hist = []
    for i in range(100):
        # test reset
        obs, infos = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)
        assert all([agent_id in obs.keys() for agent_id in env._agents])
        pad_width = env._pred_vision
        assert all([observation[int(window_l *window_l*1.5)] == 1 for observation in obs.values()])
        assert all([observation.shape == env._observation_spaces[agent_id].shape \
                    for agent_id, observation in obs.items()])
        assert np.sum(env.state()[:, :, env.PREDATOR_CHANNEL]) == len(env._agents), "Predator count should be equal to number of agents"
        assert np.sum(env.state()[:, :, env.PREY_CHANNEL]) == env._nprey, "Prey count should be equal to nprey"
        
        # test global  state 
        state = env.state()
        assert isinstance(state, np.ndarray)
        assert state.shape == env._state_space
        assert all([state[agent.position[0], agent.position[1], env.PREDATOR_CHANNEL] == 1 \
                    for agent in env._predators.values()])
        # assert sum of ones in the state is equal to number of prey
        # assert sum of ones in the state is equal to number of predators
        pd = env._pred_vision
        # print all the positions of the preys 
        global_state = env.state()
        prey_positions = np.argwhere(global_state[:, :, env.PREY_CHANNEL] == 1)
        # os.system("clear")
        while env.agents:
            os.system("clear")
            print(env.render())
            time.sleep(0.3)
            # actions = {agent_id: chaser.get_action(obs[agent_id]) \
            #         for agent_id in env.agents}
            actions = {
                "predator_0": chaser_agent.get_action(obs["predator_0"]),
                "predator_1": chaser_agent.get_action(obs["predator_1"]),
            }
            # print(f"Actions: {env.ACTION_TO_STR[actions['predator_0']]} {env.ACTION_TO_STR[actions['predator_1']]}")
            # print(f"Positions: {env._predators['predator_0'].position} {env._predators['predator_1'].position}")
            obs, rewards, terminated, truncated, infos = env.step(actions)

            # assert all([observation[pd, pd, 1] == 1 for observation in obs.values()])
            # assert sum of 1s in the predator channel is equals len(env._agents)
            # assert sum of 1s in the prey channel is equals len(env._nprey - env._kill_count)
            global_state = env.state()
            assert(np.sum(global_state[:, :, 2]) == env._nprey - env._kills)

        observations, sum_rewards, terminated, truncated, infos = env.last()
        # print(f"Game ended in {env._step_count} steps") 
        # print(f"Total kills: {env._kills}")
        # print(f"Total rewards: {env._rewards_sum}")
        if env._reward_type == "type_2":
            assert env._kills <= env._nprey, "All preys should be dead"
            # assert sum(sum_rewards.values()) ==\
            #       ((env._assists*1.5)+\
            #        ((env._kills - env._assists)*1)),\
            #           "Total rewards should be equal to number of preys killed"
        else:
            # sum of rewards is == kills 
            assert env._kills == sum(sum_rewards.values()), "All preys should be dead"
            assert env._kills == env._nprey, "All preys should be dead"
        assert env.is_terminal, "Environment should be terminal"
        assists_hist.append(env._assists)
        reward_hist.append(sum_rewards)
        steps_hist.append(env._step_count)
        kills_hist.append(env._kills)   
        # print(f"Game {i} ended in {env._step_count} steps")

    import pandas as pd 
    print(f"Average reward: {pd.DataFrame(reward_hist).mean()}")
    print(f"Average steps: {np.mean(steps_hist)}")
    print(f"Average assists: {np.mean(assists_hist)}")
    print(f"Average kills: {np.mean(kills_hist)}")