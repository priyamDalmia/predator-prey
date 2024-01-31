import gymnasium as gym
import os 
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from matplotlib.pylab import f
from pettingzoo.utils.env import ParallelEnv, ObsType, ActionType, AgentID
import numpy as np
import pandas as pd
import random
import time 
import math 

TEST = True 


class wolfpack_discrete(ParallelEnv):
    """Discerete Space (2D) Predator Prey Environment
    Predators and Preys are randomly placed on a 2D grid.
    Predators must capture Prey.
    Predators are the agents.
    Preys are either fixed or move randomly.
    Game ends when all Preys are captured or max_cycles is reached.
    """

    NUM_CHANNELS = 3
    GROUND_CHANNEL = 0
    PREDATOR_CHANNEL = 1
    PREY_CHANNEL = 2

    # GRID (0, 0) : UPPER LEFT, (N , N) : LOWER RIGHT.
    NUM_ACTIONS = 5
    ACTIONS = {
        0: lambda pos_x, pos_y: (pos_x, pos_y),  # STAY
        1: lambda pos_x, pos_y: (pos_x - 1, pos_y),  # LEFT
        2: lambda pos_x, pos_y: (pos_x + 1, pos_y),  # RIGHT
        3: lambda pos_x, pos_y: (pos_x, pos_y + 1),  # DOWN
        4: lambda pos_x, pos_y: (pos_x, pos_y - 1),  # UP
    }

    ACTION_TO_DIR = {
        0: [(0, 0), (0,0)],
        1: [(-1, 0), (0,0)],
        2: [(1, 0), (0,0)],
        3: [(0, 0), (0, 1)],
        4: [(0, 0), (0, -1)],
    }

    STR_TO_ACTION = {
        "STAY": 0,
        "UP": 1,
        "DOWN": 2,
        "RIGHT": 3,
        "LEFT": 4,
    }

    ACTION_TO_STR = {
        0: "STAY",
        1: "UP",
        2: "DOWN",
        3: "RIGHT",
        4: "LEFT",
    }

    # helper function to get the kill area slice
    kill_area_slice = lambda state, position, channel: state[
        position[0] - 2 : position[0] + 3, position[1] - 2 : position[1] + 3, channel
    ].copy()

    def __init__(
        self,
        map_size: int = 20,
        max_cycles: int = 100,
        npred: int = 2,
        nprey: int = 2,
        pred_vision: int = 2,
        reward_lone: float = 1.0,
        reward_team: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.npred = npred
        self.nprey = nprey
        self.pred_vision = pred_vision
        self.reward_lone = reward_lone
        self.reward_team = reward_team
        # build base game state here
        self.map_pad = self.pred_vision + 1
        arr = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        # define what the observation looks like
        self._base_state = np.dstack(
            [
                # a channel with 1s padded = ground channel
                np.pad(arr, self.map_pad, "constant", constant_values=1),
                # a channel with 0s padded = unit (predator channel)
                np.pad(arr, self.map_pad, "constant", constant_values=0),
                # a channel with 0s padded = unit (prey channel)
                np.pad(arr, self.map_pad, "constant", constant_values=0),
            ]
        )
        self._state_space = self._base_state.shape
        self._state_space_center = self.map_pad + self.map_size // 2

        window_l = 2 * self.pred_vision + 1
        self._observation_space = Box(
            low=0,
            high=1,
            shape=(window_l, window_l, self.NUM_CHANNELS),
            dtype=np.int32,
        )
        self._action_space = Discrete(self.NUM_ACTIONS)
        self._observation_spaces = {
            agent: self._observation_space for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }

        self.take_action = lambda position, action: self.ACTIONS[action](*position)
        self._metadata = {
            "name": "discrete_pp_v0",
            "render.modes": ["human", "rgb_array"],
            "map_size": self.map_size,
            "reward_lone": self.reward_lone,
            "reward_team": self.reward_team,
            "max_cycles": self.max_cycles,
            "npred": self.npred,
            "nprey": self.nprey,
            "pred_vision": self.pred_vision,
        }

        if self.npred > 2:
            raise NotImplementedError(
                "Only 2 predators supported for now. Redo Reward func2 for compatibility"
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Resets the environment.

        And returns a dictionary of observations (keyed by the agent name)
        """
        observations = rewards = terminated = truncated = infos = {}
        self._agents = self.possible_agents
        self._nprey = self.nprey
        self._observations = {}
        self._is_terminal = False
        self._game_step = 0
        self._game_state = self._base_state.copy()
        # store game history in a pd data frame
        columns = [
            "step",
            "done",
            "total_rewards",
            "total_kills",
            "total_assists",
            "npred",
            "nprey",
        ] + [
            f"predator_{i}_{k}"
            for k in ["rewards", "kills", "assists", "position", "action", "done"]
            for i in range(self.npred)
        ]
        self._game_history = pd.DataFrame(
            np.zeros([self.max_cycles+1, len(columns)]), columns=columns, dtype=np.object_
        )
        start_positions = set()
        # k = self.pred_vision * 2 + 1
        # k = k if k < 8 else 5
        # # TODO warning that that quad spawn only works for map size 15 or 20
        # for x_lim, y_lim in list(
        #     (i, j)
        #     for i in range(k, self.map_size + k, k)
        #     for j in range(k, self.map_size + k, k)
        # ):
        #     x_lim += self.pred_vision
        #     y_lim += self.pred_vision
        #     # sample x, y positions from the quadrant
        #     while True:
        #         x = np.random.randint(x_lim - k, x_lim)
        #         y = np.random.randint(y_lim - k, y_lim)
        #         if (x, y) not in start_positions:
        #             start_positions.add((x, y))
        #             break

        num_pred_prey = self.npred + self.nprey
        while len(list(start_positions)) < num_pred_prey:
            x = np.random.randint(self.map_pad, self.map_pad + self.map_size)
            y = np.random.randint(self.map_pad, self.map_pad + self.map_size)
            if (x, y) not in start_positions:
                start_positions.add((x, y))

        # add sanity check to ensure that all start positions are in the
        # playable are of the global state, that is accounting for the padding.
        assert all(
            [
                0 + self.map_pad <= x < self.map_pad + self.map_size + 1
                and 0 + self.map_pad <= y < self.map_pad + self.map_size + 1
                for x, y in start_positions
            ]
        ), "Start positions are not in the playable area of the global state."

        self.agent_positions = {}
        for agent_id in self._agents:
            spawn_at = start_positions.pop()
            self.agent_positions[spawn_at] = agent_id
            self._game_state[spawn_at[0], spawn_at[1], self.PREDATOR_CHANNEL] = 1
            rewards[agent_id] = 0
            terminated[agent_id] = False
            truncated[agent_id] = False
            infos[agent_id] = dict(
                assists=0,
                kills=0,
            )

        assert (
            self._game_state[:, :, self.PREDATOR_CHANNEL].sum() == self.npred
        ), "Number of predators should be equal to number of agents"

        for _ in range(self.nprey):
            position = start_positions.pop()
            self._game_state[position[0], position[1], self.PREY_CHANNEL] = 1

        observations = {
            agent_id: self.get_observation(agent_id) for agent_id in self._agents
        }
        return observations, infos
    
    def predators_around_kill(self, predator_position, prey_position):
        curr_assists = []
        curr_kills = []
        for pos, agent_id in self.agent_positions.items():
            if pos == predator_position:
                curr_kills.append(agent_id)
            else:
                if math.dist(pos,  predator_position) <= 2:
                    curr_assists.append(agent_id)
                elif math.dist(pos, prey_position) <= 2:
                    curr_assists.append(agent_id)
        return curr_kills, curr_assists

    def step(
        self, actions: dict[AgentID, int]
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
        assert set(self._agents) == set(
            actions.keys()
        ), f"Incorrect action dict,\ngiven: {actions.keys()}, \nexpected: {self._agents}"
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        last_positions = {}
        curr_kills = []
        curr_assists = []
        actions_list = list(actions.items())
        random.shuffle(actions_list)
        for agent_id, action in actions_list:
            # get position of the predator
            rewards[agent_id] = 0
            agent_position = None
            for position, _id in self.agent_positions.items():
                if _id == agent_id:
                    agent_position = position
                    break

            if agent_position is None:
                raise ValueError(f"Agent {agent_id} is not alive or does not exist.")

            last_positions[agent_id] = agent_position
            next_position = self.take_action(agent_position, action)
            # if new position is wall or obstacle; do not take action and add zero reward
            if (
                self._game_state[
                    next_position[0], next_position[1], self.GROUND_CHANNEL:self.PREY_CHANNEL
                ].sum()
                == 1
            ):
                continue
            elif (
                self._game_state[next_position[0], next_position[1], self.PREY_CHANNEL]
                == 1
            ):
                # kill the prey
                # give reward acording to distribution
                kills, assists = self.predators_around_kill(
                    agent_position, next_position
                )
                curr_kills.extend(kills)
                curr_assists.extend(assists)
                self._game_state[agent_position[0], agent_position[1], self.PREDATOR_CHANNEL] = 0
                self.agent_positions.pop(agent_position)  # move the agent
                self._game_state[next_position[0], next_position[1], self.PREY_CHANNEL] = 0
                self._game_state[next_position[0], next_position[1], self.PREDATOR_CHANNEL] = 1
                self.agent_positions[next_position] = agent_id
            else:
                # update to new position
                self._game_state[agent_position[0], agent_position[1], self.PREDATOR_CHANNEL] = 0
                self.agent_positions.pop(agent_position)
                self._game_state[next_position[0], next_position[1], self.PREDATOR_CHANNEL] = 1
                self.agent_positions[next_position] = agent_id

        # update rewards, log kills and assists, log game history
        self._nprey -= len(curr_kills)
        for agent_id in list(self._agents):
            if self.is_terminal:
                self._agents.remove(agent_id)
                terminated[agent_id] = True
                truncated[agent_id] = True
            else:
                terminated[agent_id] = False
                truncated[agent_id] = False

            if agent_id in curr_kills:
                if len(curr_assists) > 1:
                    rewards[agent_id] = self.reward_team + rewards.get(agent_id, 0)
                else:
                    rewards[agent_id] = self.reward_lone + rewards.get(agent_id, 0)
            
            if agent_id in curr_assists:
                rewards[agent_id] = self.reward_team + rewards.get(agent_id, 0)

            self._game_history.loc[self._game_step, f"{agent_id}_rewards"] = rewards[
                agent_id
            ]
            self._game_history.loc[self._game_step, f"{agent_id}_kills"] = (
                1 if agent_id in curr_kills else 0
            )
            self._game_history.loc[self._game_step, f"{agent_id}_assists"] = (
                1 if agent_id in curr_assists else 0
            )
            self._game_history.loc[
                self._game_step, f"{agent_id}_position"
            ] = str(last_positions[agent_id])
            
            self._game_history.loc[
                self._game_step, f"{agent_id}_action"
            ] = actions[agent_id]
            self._game_history.loc[self._game_step, f"{agent_id}_done"] = int(
                terminated[agent_id] or truncated[agent_id]
            )
        self._game_history.loc[self._game_step, "step"] = self._game_step
        self._game_history.loc[self._game_step, "done"] = int(self.is_terminal)
        self._game_history.loc[self._game_step, "total_rewards"] = sum(rewards.values())
        self._game_history.loc[self._game_step, "total_kills"] = len(curr_kills)
        self._game_history.loc[self._game_step, "total_assists"] = len(curr_assists)
        self._game_step += 1

        # update infos
        next_observations = {agent_id: self.get_observation(agent_id)\
                              for agent_id in self._agents}
        infos['__all__'] = dict() 
        return next_observations, rewards, terminated, truncated, infos

    @property
    def is_terminal(
        self,
    ) -> bool:
        if (
            self._game_step > self.max_cycles-1
            or len(self._agents) == 0
            or self._game_state[:, :, self.PREY_CHANNEL].sum() == 0
        ):
            self._is_terminal = True
        return self._is_terminal

    @property
    def agents(self):
        # if attribute does not exist, raise AttributeError ask to reset first
        if not hasattr(self, "_agents"):
            raise AttributeError("Must call reset() first after env initialization")
        return self._agents

    @property
    def possible_agents(self):
        return list(set([f"predator_{i}" for i in range(self.npred)]))

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

        # get a shifed view of the observation based on last action 
        # if step is zero; chose random direction
        # if last action is zero; get last non zero action 
        for position, agent in self.agent_positions.items():
            if agent_id == agent:
                break
        
        last_action = self._game_history.loc[self._game_step, f'{agent_id}_action']
        direction = self.ACTION_TO_DIR[last_action]
        pos_x, pos_y = position
        offset_x, offset_y = direction 
        window_l = 2 * self.pred_vision + 1
        observation = self._game_state[
            (pos_x - self.pred_vision + offset_x[0]): (pos_x + self.pred_vision + offset_x[1]+ 1),
            (pos_y - self.pred_vision + offset_y[0]): (pos_y + self.pred_vision + offset_y[1] + 1),
            :,
        ].copy()
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
        lims = self.map_pad
        render_array = self._game_state[
            lims : (lims + self.map_size), lims : (lims + self.map_size), 0
        ].copy()
        render_array = render_array.astype(str)
        render_array = np.char.replace(render_array, "0", ".")
        render_array = render_array.astype("U11")

        for x, y in zip(*np.where(self._game_state[:, :, self.PREDATOR_CHANNEL])):
            agent_id = self.agent_positions[(x,y)]
            render_array[x - lims, y - lims] = f"P{agent_id[-1]}"

        for x, y in zip(*np.where(self._game_state[:, :, self.PREY_CHANNEL])):
            render_array[x - lims, y - lims] = "x"

        return render_array

    def close(self):
        """Closes the rendering window."""
        pass

    def state(self) -> np.ndarray:
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        return self._game_state.copy()

    def last(self):
        return ()
        #     self._observations.copy(),
        #     self._rewards_sum.copy(),
        #     self._terminated.copy(),
        #     self._truncated.copy(),
        #     self._infos.copy(),
        # )

    @property
    def unwrapped(self) -> ParallelEnv:
        return self


if __name__ == "__main__":
    config = dict(
        map_size=10,
        max_cycles=100,
        npred=2,
        pred_vision=3,
        nprey=2,
        reward_lone=1.0,
        reward_team=1.0,
        render_mode=None,
    )

    env = wolfpack_discrete(**config)
    # fixed_agent = FixedSwingAgent(env)
    # follower_agent = FollowerAgent(env)
    from discrete_pp_v1 import ChaserAgent
    chaser_agent = ChaserAgent(env)
    print(f"Env name: {env.__name__()}, created!")

    # test action and observation spaces
    assert isinstance(env.observation_spaces, dict)
    assert isinstance(env.action_spaces, dict)

    # if map size is 20 and vision size is 2,
    # then the observation space should be 5x5x3
    window_l = 2 * env.pred_vision + 1
    assert all(
        [
            env.observation_spaces[agent_id].shape
            == (window_l, window_l, env.NUM_CHANNELS)
            for agent_id in env.possible_agents
        ]
    )
    assert all(
        [
            env.action_spaces[agent_id].n == env.NUM_ACTIONS
            for agent_id in env.possible_agents
        ]
    )

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
        pad_width = env.pred_vision
        assert all(
            [
                observation[pad_width, pad_width, env.PREDATOR_CHANNEL] == 1
                for observation in obs.values()
            ]
        )
        assert all(
            [
                observation.shape == env._observation_spaces[agent_id].shape
                for agent_id, observation in obs.items()
            ]
        )
        assert np.sum(env.state()[:, :, env.PREDATOR_CHANNEL]) == len(
            env._agents
        ), "Predator count should be equal to number of agents"
        assert (
            np.sum(env.state()[:, :, env.PREY_CHANNEL]) == env._nprey
        ), "Prey count should be equal to nprey"

        # test global  state
        state = env.state()
        assert isinstance(state, np.ndarray)
        assert state.shape == env._state_space
        assert all(
            [
                state[position[0], position[1], env.PREDATOR_CHANNEL] == 1
                for position in env.agent_positions.keys()
            ]
        )
        # assert sum of ones in the state is equal to number of prey
        # assert sum of ones in the state is equal to number of predators
        # print all the positions of the preys
        global_state = env.state()
        prey_positions = np.argwhere(global_state[:, :, env.PREY_CHANNEL] == 1)
        os.system("clear")
        while env.agents:
            # actions = {agent_id: env.action_spaces[agent_id].sample() \
            #         for agent_id in env.agents}
            actions = {
                "predator_0": chaser_agent.get_action(obs["predator_0"]),
                "predator_1": chaser_agent.get_action(obs["predator_1"]),
            }

            print(f"STEP: {env._game_step}")
            for position, agent in env.agent_positions.items():
                print(f"{agent} at {position}")
                assert env._game_state[position[0], position[1], 1] == 1, "Predator at incoorect position"

            print(env.render())
            print(f"Actions: {[(k, env.ACTION_TO_STR[v]) for k, v in actions.items()]}")
            obs, rewards, terminated, truncated, infos = env.step(actions)
            print(env.render())
            print(f"Rewards: {[(k, v) for k, v in rewards.items()]}")
            time.sleep(0.5)
            for position, agent in env.agent_positions.items():
                print(f"{agent} at {position}")
                assert env._game_state[position[0], position[1], 1] == 1, "Predator at incoorect position"
            # assert sum of 1s in the predator channel is equals len(env._agents)
            # assert sum of 1s in the prey channel is equals len(env._nprey - env._kill_count)
            os.system("clear")
            global_state = env.state()
            assert np.sum(global_state[:, :, 2]) == env._nprey

        # observations, sum_rewards, terminated, truncated, infos = env.last()
        
        total_assists = env._game_history['total_assists'].sum()
        total_rewards = env._game_history['total_rewards'].sum()
        total_kills = env._game_history['total_kills'].sum()
        p0_assists = env._game_history['predator_0_assists'].sum()
        p0_kills = env._game_history['predator_0_kills'].sum()
        p0_rewards = env._game_history['predator_0_rewards'].sum()
        p1_assists = env._game_history['predator_1_assists'].sum()
        p1_kills = env._game_history['predator_1_kills'].sum()
        p1_rewards = env._game_history['predator_1_rewards'].sum()

        assert total_assists == p0_assists + p1_assists, "incorrect assists"
        assert total_rewards == p0_rewards + p1_rewards, "incorrect rewards"
        assert total_kills == p0_kills + p1_kills, "incorrect kills"
    print(f"Average reward: {pd.DataFrame(reward_hist).mean()}")
    print(f"Average steps: {np.mean(steps_hist)}")
    print(f"Average assists: {np.mean(assists_hist)}")
    print(f"Average kills: {np.mean(kills_hist)}")
