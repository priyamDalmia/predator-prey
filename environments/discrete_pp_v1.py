from curses import meta
import dis
from logging import warning
from math import sqrt
import os 
import sys
import time
import math
from torch import rand
path1 = '/home/priyam/projects/predator-prey'
sys.path.append(path1)
sys.path.append(os.path.join(path1, "environments"))
import random
    # TODO move and build tests s 
import numpy as np 
from common import Agent
from typing import Any, Dict
import gymnasium 
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv, ObsType, ActionType, AgentID

class discrete_pp_v1(ParallelEnv):
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
        # step penalty 
        self._step_penalty = abs(kwargs.get("step_penalty", 0.0))
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
            shape=(window_l, window_l, self.NUM_CHANNELS),
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
    
    def reward_dist_1(self, rewards, agent_id, agent_position, kill_position):
        kill_area_1 = discrete_pp_v1.kill_area_slice(
            self._global_state, kill_position, self.PREDATOR_CHANNEL)
        kill_area_2 = discrete_pp_v1.kill_area_slice(
            self._global_state, agent_position, self.PREDATOR_CHANNEL)
        
        if (kill_area_1.sum()+kill_area_2.sum()) > 2:
            self._assists += 1
            for _id in self._possible_agents:
                if agent_id != _id:
                    self._assists_by_id[_id] = 1 + self._assists_by_id.get(_id, 0)
        rewards[agent_id] = 1 + rewards.get(agent_id, 0)
        return rewards

    def reward_dist_2(self, rewards, agent_id, agent_position, kill_position):  
        kill_area_1 = discrete_pp_v1.kill_area_slice(
            self._global_state, kill_position, self.PREDATOR_CHANNEL)
        kill_area_2 = discrete_pp_v1.kill_area_slice(
            self._global_state, agent_position, self.PREDATOR_CHANNEL)
        
        if (kill_area_1.sum()+kill_area_2.sum()) > 2:
            self._assists += 1
            for _id in self._possible_agents:
                rewards[_id] = 0.80 + rewards.get(_id, 0)
                if agent_id != _id:
                    self._assists_by_id[_id] = 1 + self._assists_by_id.get(_id, 0)
            return rewards
        else:
            rewards[agent_id] = 1 + rewards.get(agent_id, 0)
            return rewards
    
    def reward_dist_3(self, rewards, agent_id, agent_position, kill_position):  
        kill_area_1 = discrete_pp_v1.kill_area_slice(
            self._global_state, kill_position, self.PREDATOR_CHANNEL)
        kill_area_2 = discrete_pp_v1.kill_area_slice(
            self._global_state, agent_position, self.PREDATOR_CHANNEL)
        
        if (kill_area_1.sum()+kill_area_2.sum()) > 2:
            self._assists += 1
            for _id in self._possible_agents:
                rewards[_id] = 1.2 + rewards.get(_id, 0)
                if agent_id != _id:
                    self._assists_by_id[_id] = 1 + self._assists_by_id.get(_id, 0)
            return rewards
        else:
            rewards[agent_id] = 1 + rewards.get(agent_id, 0)
            return rewards
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Resets the environment.

        And returns a dictionary of observations (keyed by the agent name)
        """
        self._observations = {}
        self._infos = {}
        self._rewards = {}
        self._terminated = {}
        self._truncated = {}
        self._rewards_sum = {}
        self._kills = 0
        self._assists = 0
        self._kills_by_id = {}
        self._assists_by_id = {}
        self._agents = self._possible_agents.copy()
        self._predators = {}
        self._step_count = 0
        self._global_state = self._base_state.copy()
        start_positions = set()
        k = self._pred_vision * 2 + 1
        k = k if k < 8 else 5
        # TODO warning that that quad spawn only works for map size 15 or 20
        for x_lim, y_lim in list(
            (i, j) for i in range(k, self._map_size+k, k) for j in range(k, self._map_size+k, k)):
            x_lim += self._pred_vision
            y_lim += self._pred_vision
            # sample x, y positions from the quadrant
            while True:
                x = np.random.randint(x_lim - k, x_lim)
                y = np.random.randint(y_lim - k, y_lim)
                if (x, y) not in start_positions:
                    start_positions.add((x, y))
                    break

        num_pred_prey = self._npred + self._nprey
        while len(list(start_positions)) < num_pred_prey:
            x = np.random.randint(self._pred_vision, self._pred_vision+self._map_size)
            y = np.random.randint(self._pred_vision, self._pred_vision+self._map_size)
            if (x, y) not in start_positions:
                start_positions.add((x, y))

        # add sanity check to ensure that all start positions are in the 
        # playable are of the global state, that is accounting for the padding.
        assert all([0+self._pred_vision<=x<self._pred_vision+self._map_size+1 and 
                    0+self._pred_vision<=y<self._pred_vision+self._map_size+1 
                        for x, y in start_positions]),\
                         "Start positions are not in the playable area of the global state."
        
        for agent_id in self._agents:
            agent = Agent(agent_id)
            spawn_at = start_positions.pop()
            agent(spawn_at)
            self._global_state[spawn_at[0], spawn_at[1], self.PREDATOR_CHANNEL] = 1
            self._predators[agent_id] = agent
            self._rewards[agent_id]= float(0)
            self._terminated[agent_id] = False
            self._truncated[agent_id] = False
            self._kills_by_id[agent_id] = 0
            self._assists_by_id[agent_id] = 0
            self._infos[agent_id] = dict()
        
        assert self._global_state[:, :, self.PREDATOR_CHANNEL].sum() == self._npred, "Number of predators should be equal to number of agents"

        for _ in range(self._nprey):
            position = start_positions.pop()
            self._global_state[position[0], position[1], self.PREY_CHANNEL] = 1

        self._observations = {agent_id: self.get_observation(agent_id) for agent_id in self._agents}   
        self._rewards_sum = self._rewards.copy() 
        return self._observations.copy(), self._infos.copy()    
    
    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.
        """
        observations = {}
        infos = {}
        rewards = {}
        terminated = self._terminated.copy() 
        truncated = self._truncated.copy()
        assert set(self._agents) == set(actions.keys()), f"Incorrect action dict,\ngiven: {actions.keys()}, \nexpected: {self._agents}"

        if self._prey_type == "random":
            raise NotImplementedError("Random moving prey")

        actions_list = list(actions.items())
        random.shuffle(actions_list) 
        for agent_id, action in actions_list:
            # get position of the predators 
            agent = self._predators[agent_id]
            assert agent.is_alive and not self.is_terminal, f"Dead agent in self._agents list for game!"
            position = agent.position 
            next_position = self.take_action(position, action)

            # if new position is wall; do not take action
            if self._global_state[next_position[0], next_position[1], self.GROUND_CHANNEL] == 1:
                rewards[agent_id] = rewards.get(agent_id, 0)
                rewards[agent_id] = rewards[agent_id] - self._step_penalty
                # if wall, do not move or do anything.
                # and reward is zero 
            elif self._global_state[next_position[0], next_position[1], self.PREY_CHANNEL] == 1:
                # elif new position on prey 
                # kill the prey 
                # give reward acording to distribution
                self._kills += 1
                self._kills_by_id[agent_id] = 1 + self._kills_by_id.get(agent_id, 0)
                rewards = self._reward_func(rewards, agent_id, position, next_position)
                self._global_state[next_position[0], next_position[1], self.PREY_CHANNEL] = 0
                # move the predator 
                agent.move(next_position)
            else:
                # update to new position 
                # and reward is zero 
                rewards[agent_id] = rewards.get(agent_id, 0)
                rewards[agent_id] = rewards[agent_id] - self._step_penalty
                agent.move(next_position)
        
        arr = np.zeros(
            (self._map_size, self._map_size),
            dtype = np.int32)
        self._global_state[:, :, self.PREDATOR_CHANNEL] =  np.pad(arr, self._map_pad, 'constant', constant_values=0)
        for agent_id in list(self._agents):
            agent = self._predators[agent_id]
            self._global_state[agent.position[0], agent.position[1], self.PREDATOR_CHANNEL] = 1
            self._rewards_sum[agent_id] += rewards.get(agent_id, 0)
            if self._step_count == self._max_cycles:
                truncated[agent_id] = True
                terminated[agent_id] = False
                self._agents.remove(agent_id)
            elif self._kills == self._nprey:
                truncated[agent_id] = False
                terminated[agent_id] = True
                self._agents.remove(agent_id)
            else:
                infos[agent_id] = dict(
                    assists = self._assists_by_id[agent_id],
                    kills = self._kills_by_id[agent_id],
                )

        infos['__all__'] = dict(
            kills = self._kills,
            assists = self._assists,
            kills_by_id = self._kills_by_id.copy(),
            assists_by_id = self._assists_by_id.copy(),
        )
        self._observations = {agent_id: self.get_observation(agent_id) for agent_id in self._agents}
        self._rewards = rewards.copy()
        self._terminated = terminated.copy()
        self._truncated = truncated.copy()
        self._step_count += 1
        return self._observations.copy(), rewards, terminated, truncated, infos.copy()

    @property
    def is_terminal(self,) -> bool:
        #TODO check if all prey are dead
        return all(list(self._terminated.values())) or all(list(self._truncated.values()))
    
    @property
    def agents(self):
        # if attribute does not exist, raise AttributeError ask to reset first
        if not hasattr(self, "_agents"):
            raise AttributeError("Must call reset() first after env initialization")
        return self._agents

    @property
    def possible_agents(self):
        return self._possible_agents 
    
    @property
    def observation_spaces(self):
        return self._observation_spaces
    
    @property
    def action_spaces(self):
        return self._action_spaces
    
    @property
    def metadata(self):
        return self._metadata
    
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
        return observation

    def __name__(self):
        return "discrete_pp_v1"
    
    def render(self) -> None | np.ndarray | str | list:
        """Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        lims = self._pred_vision
        state = self.state()
        render_array = state[lims:(lims+self._map_size), lims:(lims+self._map_size), 0].copy()
        render_array = render_array.astype(str)
        render_array = np.char.replace(render_array, '0', '.')
        render_array = render_array.astype('U11')

        for x,y in zip(*np.where(state[:, :, self.PREDATOR_CHANNEL])):
            render_array[x-lims, y-lims] = 'P'
        
        for x,y in zip(*np.where(state[:, :, self.PREY_CHANNEL])):
            render_array[x-lims, y-lims] = 'x'
        
        return render_array.T 


    def close(self):
        """Closes the rendering window."""
        pass

    def state(self) -> np.ndarray:
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        return self._global_state.copy()

    def last(self):
        return (
            self._observations.copy(),
            self._rewards_sum.copy(),
            self._terminated.copy(),
            self._truncated.copy(),
            self._infos.copy(),
        )

    @property
    def unwrapped(self) -> ParallelEnv:
        return self

class FixedSwingAgent:
    def __init__(self, env=None) -> None:
        self.direction = random.choice(["LEFT", "RIGHT"])
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D 
            a = int(math.sqrt(observation.shape[0]/3))
            observation = observation.reshape(a,a,3)
        center = observation.shape[0] // 2
        observation = observation.T
        # if close of the left wall, change direction and move 
        if observation[0, center, :center].sum() > 1:
            self.direction = "RIGHT"
            return discrete_pp_v1.STR_TO_ACTION[self.direction]
        elif observation[0, center, center:].sum() > 1:
            self.direction = "LEFT"
            return discrete_pp_v1.STR_TO_ACTION[self.direction]
        else:
            if np.random.random() < 0.8:
                return discrete_pp_v1.STR_TO_ACTION[self.direction]
            else:
                # if close to the top wall, move down
                if observation[0, :center, center].sum() > 1:
                    return discrete_pp_v1.STR_TO_ACTION["DOWN"]
                elif observation[0, center:, center].sum() > 1:
                    return discrete_pp_v1.STR_TO_ACTION["UP"]
                else:
                    return discrete_pp_v1.STR_TO_ACTION[random.choice(["UP", "DOWN"])]
    
    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None

    def get_initial_state(self):
        return 0

class FollowerAgent:
    """
    If predator in vision, takes a step in its direction 
    else, randomly moves
    """
    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D 
            a = int(math.sqrt(observation.shape[0]/3))
            observation = observation.reshape(a,a,3)
        center = observation.shape[0] // 2

        if observation[:, :, discrete_pp_v1.PREDATOR_CHANNEL].sum() > 1:
            for position in zip(*np.where(observation[:, :, discrete_pp_v1.PREDATOR_CHANNEL])):
                if position[0] == center and position[1] == center:
                    continue
                elif position[0] < center:
                    return discrete_pp_v1.STR_TO_ACTION["LEFT"]
                elif position[0] > center:
                    return discrete_pp_v1.STR_TO_ACTION["RIGHT"]
                elif position[1] < center:
                    return discrete_pp_v1.STR_TO_ACTION["UP"]
                elif position[1] > center:
                    return discrete_pp_v1.STR_TO_ACTION["DOWN"]
        else:
            return discrete_pp_v1.STR_TO_ACTION[random.choice(["UP", "DOWN", "LEFT", "RIGHT"])]
    
    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None
    
    def get_initial_state(self):
        return 0 

class ChaserAgent:
    """
    If prey in vision, takes a step in its direction (closet prey)
    else, randomly moves
    """
    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D 
            a = int(math.sqrt(observation.shape[0]/3))
            observation = observation.reshape(a,a,3)

        center = observation.shape[0] // 2

        if observation[:, :, discrete_pp_v1.PREY_CHANNEL].sum() > 0:
            positions = list(zip(*np.where(observation[:, :, discrete_pp_v1.PREY_CHANNEL])))
            distance = lambda pos: abs(pos[0] - center) + abs(pos[1] - center)
            positions.sort(key=distance)
            for position in positions: 
                if position[0] < center:
                    return discrete_pp_v1.STR_TO_ACTION["LEFT"]
                elif position[0] > center:
                    return discrete_pp_v1.STR_TO_ACTION["RIGHT"]
                elif position[1] < center:
                    return discrete_pp_v1.STR_TO_ACTION["UP"]
                else:
                    return discrete_pp_v1.STR_TO_ACTION["DOWN"]
        else:
            return discrete_pp_v1.STR_TO_ACTION[random.choice(["UP", "DOWN", "LEFT", "RIGHT"])]
    
    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None
    
    def get_initial_state(self):
        return 0 

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

    env = discrete_pp_v1(**config)
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
    assert all([env.observation_spaces[agent_id].shape == \
                (window_l, window_l, env.NUM_CHANNELS) \
                for agent_id in env.possible_agents]) 
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
        assert all([observation[pad_width, pad_width, env.PREDATOR_CHANNEL] == 1 for observation in obs.values()])
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
            # os.system("clear")
            # print(env.render())
            # actions = {agent_id: chaser.get_action(obs[agent_id]) \
            #         for agent_id in env.agents}
            actions = {
                "predator_0": chaser_agent.get_action(obs["predator_0"]),
                "predator_1": chaser_agent.get_action(obs["predator_1"]),
            }
            # print(f"Actions: {env.ACTION_TO_STR[actions['predator_0']]} {env.ACTION_TO_STR[actions['predator_1']]}")
            # print(f"Positions: {env._predators['predator_0'].position} {env._predators['predator_1'].position}")
            obs, rewards, terminated, truncated, infos = env.step(actions)

            assert all([observation[pd, pd, 1] == 1 for observation in obs.values()])
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