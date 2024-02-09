from typing import Any
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


class AgressiveAgent:
    """
    If predators in direct line of sight, shoots
    If prey in vision, takes a step in its direction (closet prey)
    else, randomly moves
    """

    def __init__(self, env=None) -> None:
        pass

    def get_action(self, observation):
        if len(observation.shape) != 3:
            # reshape 1D observation back into 3D
            a = int(math.sqrt(observation.shape[0] / 3))
            observation = observation.reshape(a, a, 3)

        center = list(zip(*np.where(observation[:, :, 3] == 1)))[0]
        if observation[:, :, gather_v1.PREDATOR_CHANNEL].sum() > 1:
            c = observation.shape[0] // 2
            if center[0] > c:
                if (
                    observation[
                        : center[0], center[1], gather_v1.PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return gather_v1.STR_TO_ACTION["SHOOT"]
            elif center[0] < c:
                if (
                    observation[
                        center[0] + 1 :, center[1], gather_v1.PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return gather_v1.STR_TO_ACTION["SHOOT"]
            elif center[1] > c:
                if (
                    observation[
                        center[0], : center[1], gather_v1.PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return gather_v1.STR_TO_ACTION["SHOOT"]
            elif center[1] < c:
                if (
                    observation[
                        center[0], center[1] + 1 :, gather_v1.PREDATOR_CHANNEL
                    ].sum()
                    == 1
                ):
                    return gather_v1.STR_TO_ACTION["SHOOT"]

        if observation[:, :, gather_v1.PREY_CHANNEL].sum() > 0:
            positions = list(zip(*np.where(observation[:, :, gather_v1.PREY_CHANNEL])))
            distance = lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            positions.sort(key=distance)
            for position in positions:
                if position[0] > center[0]:
                    return gather_v1.STR_TO_ACTION["DOWN"]
                elif position[0] < center[0]:
                    return gather_v1.STR_TO_ACTION["UP"]
                elif position[1] < center[1]:
                    return gather_v1.STR_TO_ACTION["LEFT"]
                elif position[1] > center[1]:
                    return gather_v1.STR_TO_ACTION["RIGHT"]
                break
        return np.random.randint(0, 8)

    def compute_single_action(self, observation, *args, **kwargs):
        return self.get_action(observation), None, None

    def get_initial_state(self):
        return 0


class Predator:
    def __init__(self, name, observation_space, action_space):
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self._is_alive = False
        self._position = None
        self._direction = None
        self._is_stunned = False
        self._stun_count = 0

    def __call__(self, spawn_position: tuple):
        self._is_alive = True
        self._position = spawn_position
        self._direction = np.random.randint(1, 5)
        self._is_stunned = False
        self._stun_count = 0

    def __repr__(self):
        direction_str = gather_v1.ACTION_TO_STR[self.direction + 4][6:]
        return f"{self.name} at {self.position} facing {direction_str}"

    @property
    def is_alive(self):
        return self._is_alive

    @property
    def position(self):
        if self._position is None:
            raise Exception(f"{self.name} is not alive!")
        return self._position

    @property
    def is_stunned(self):
        return self._is_stunned

    @property
    def direction(self):
        return self._direction

    def stun(self, stun_rate):
        self._is_stunned = True
        self._stun_count = stun_rate

    def kill(self):
        self._is_alive = False
        self._position = None

    def move(self, new_position):
        self._position = new_position

    def rotate(self, new_direction):
        self._direction = new_direction

    def step_stunned(self):
        if self._is_stunned:
            self._stun_count -= 1
            if self._stun_count == 0:
                self._is_stunned = False
        else:
            raise Exception(f"{self.name} is not stunned!")


class gather_v1(ParallelEnv):
    """Discerete Space (2D) Predator Prey Environment
    Predators and Preys are randomly placed on a 2D grid.
    Predators must capture Prey.
    Predators are the agents.
    Preys are either fixed or move randomly.
    Game ends when all Preys are captured or max_cycles is reached.
    """

    # GLOBAL STATE
    NUM_CHANNELS = 3
    GROUND_CHANNEL = 0
    PREDATOR_CHANNEL = 1
    PREY_CHANNEL = 2

    # AGENTS OBSERVE EXTRA CHANNEL
    SELF_CHANNEL = 3

    # GRID (0, 0) : UPPER LEFT, (N , N) : LOWER RIGHT.
    NUM_ACTIONS = 10
    MOVE_ACTION = {
        0: lambda pos_x, pos_y: (pos_x, pos_y),  # STAY
        1: lambda pos_x, pos_y: (pos_x - 1, pos_y),  # UP
        2: lambda pos_x, pos_y: (pos_x + 1, pos_y),  # DOWN
        3: lambda pos_x, pos_y: (pos_x, pos_y + 1),  # RIGHT
        4: lambda pos_x, pos_y: (pos_x, pos_y - 1),  # LEFT
    }

    ROTATE_ACTION = {
        5: lambda direction: 1,
        6: lambda direction: 2,
        7: lambda direction: 3,
        8: lambda direction: 4,
    }

    DIRECTION_TO_VECTOR = {
        1: [(-1, -1), (0, 0)],
        2: [(1, 1), (0, 0)],
        3: [(0, 0), (1, 1)],
        4: [(0, 0), (-1, -1)],
    }

    STR_TO_ACTION = {
        "STAY": 0,
        "UP": 1,
        "DOWN": 2,
        "RIGHT": 3,
        "LEFT": 4,
        "ROTATE_UP": 5,
        "ROTATE_DOWN": 6,
        "ROTATE_RIGHT": 7,
        "ROTATE_LEFT": 8,
        "SHOOT": 9,
    }

    ACTION_TO_STR = {
        0: "STAY",
        1: "UP",
        2: "DOWN",
        3: "RIGHT",
        4: "LEFT",
        5: "ROTATE_UP",
        6: "ROTATE_DOWN",
        7: "ROTATE_RIGHT",
        8: "ROTATE_LEFT",
        9: "SHOOT",
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
        pred_stun_rate: int = 5,
        shot_penalty: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.npred = npred
        self.nprey = nprey
        self.pred_vision = pred_vision
        self.pred_stun_rate = pred_stun_rate
        self.shot_penalty = shot_penalty
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
            low=-1 * (self.pred_stun_rate + 1),
            high=1,
            shape=(window_l, window_l, self.NUM_CHANNELS + 1),
            dtype=np.int32,
        )
        self._action_space = Discrete(self.NUM_ACTIONS)
        self._observation_spaces = {
            agent: self._observation_space for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }

        self._predators = dict()
        for agent_id in self.possible_agents:
            agent = Predator(
                agent_id,
                self._observation_spaces[agent_id],
                self._action_spaces[agent_id],
            )
            self._predators[agent_id] = agent

        self._metadata = {
            "name": "gather_v1",
            "render.modes": ["human", "rgb_array"],
            "map_size": self.map_size,
            "max_cycles": self.max_cycles,
            "npred": self.npred,
            "nprey": self.nprey,
            "pred_vision": self.pred_vision,
            "pred_stun_rate": self.pred_stun_rate,
            "shot_penalty": self.shot_penalty,
        }

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
            "total_shots",
            "total_hits",
            "npred",
            "nprey",
        ] + [
            f"predator_{i}_{k}"
            for k in ["rewards", "shots", "hits", "position", "action", "done"]
            for i in range(self.npred)
        ]
        self._game_history = pd.DataFrame(
            np.zeros([self.max_cycles + 1, len(columns)]),
            columns=columns,
            dtype=np.object_,
        )
        start_positions = set()
        while len(list(start_positions)) < (self.npred + self.nprey):
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

        for agent_id in self._agents:
            spawn_at = start_positions.pop()
            self._predators[agent_id](spawn_at)
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

        assert (
            self._game_state[:, :, self.PREY_CHANNEL].sum() == self.nprey
        ), "Number of prey should be equal to number of agents"

        observations = {
            agent_id: self.get_observation(agent_id) for agent_id in self._agents
        }
        return observations, infos

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
        shots_fired = []
        shots_hit = []
        actions_order = list(actions.items())
        random.shuffle(actions_order)
        for agent_id, action in actions_order:
            # get position of the predator
            rewards[agent_id] = 0
            agent = self._predators[agent_id]
            last_positions[agent_id] = agent.position
            if agent.position is None:
                raise ValueError(f"Action given for DEAD agent {agent_id}")

            if agent.is_stunned:
                agent.step_stunned()
                continue

            next_position, next_direction = self.take_action(
                action, agent.position, agent.direction
            )
            if action == 9:
                # fires a stringht stun shot in the facing direction
                # if hit, stun the predator for pred_stun_rate steps
                fired, hit = self.predators_in_line_of_fire(
                    agent.position, agent.direction
                )
                shots_fired.extend(fired)
                shots_hit.extend(hit)
                continue
            # if action is to rotate; else move action
            elif action > 4 and action < 9:
                agent.rotate(next_direction)
            # if new position is wall or obstacle; do not take move and add zero reward
            elif (
                self._game_state[
                    next_position[0],
                    next_position[1],
                    self.GROUND_CHANNEL : self.PREY_CHANNEL,
                ].sum()
                == 1
            ):
                continue
            # if new position is at prey; kill prey and add reward
            elif (
                self._game_state[next_position[0], next_position[1], self.PREY_CHANNEL]
                == 1
            ):
                # kill the prey
                # give reward acording to distribution
                rewards[agent_id] += 1
                self._game_state[
                    next_position[0], next_position[1], self.PREY_CHANNEL
                ] = 0
                self._game_state[
                    agent.position[0], agent.position[1], self.PREDATOR_CHANNEL
                ] = 0
                self._game_state[
                    next_position[0], next_position[1], self.PREDATOR_CHANNEL
                ] = 1
                agent.move(next_position)
            else:
                # update to new position
                self._game_state[
                    agent.position[0], agent.position[1], self.PREDATOR_CHANNEL
                ] = 0
                self._game_state[
                    next_position[0], next_position[1], self.PREDATOR_CHANNEL
                ] = 1
                agent.move(next_position)

        # update rewards, log kills and assists, log game history
        self._nprey = self._game_state[:, :, self.PREY_CHANNEL].sum()
        for agent_id in list(self._agents):
            if self.is_terminal:
                self._agents.remove(agent_id)
                self._predators[agent_id].kill()
                terminated[agent_id] = True
                truncated[agent_id] = True
            else:
                terminated[agent_id] = not self._predators[agent_id].is_alive
                truncated[agent_id] = False

            if agent_id in shots_fired:
                rewards[agent_id] = -self.shot_penalty + rewards.get(agent_id, 0)
                self._game_history.loc[self._game_step, f"{agent_id}_shots"] = 1

            if agent_id in shots_hit:
                self._game_history.loc[self._game_step, f"{agent_id}_hits"] = 1

            self._game_history.loc[self._game_step, f"{agent_id}_rewards"] = rewards[
                agent_id
            ]
            self._game_history.loc[self._game_step, f"{agent_id}_position"] = str(
                last_positions[agent_id]
            )
            self._game_history.loc[self._game_step, f"{agent_id}_action"] = actions[
                agent_id
            ]
            self._game_history.loc[self._game_step, f"{agent_id}_done"] = int(
                terminated[agent_id] or truncated[agent_id]
            )
        self._game_history.loc[self._game_step, "step"] = self._game_step
        self._game_history.loc[self._game_step, "done"] = int(self.is_terminal)
        self._game_history.loc[self._game_step, "total_rewards"] = sum(rewards.values())
        self._game_history.loc[self._game_step, "total_shots"] = len(shots_fired)
        self._game_history.loc[self._game_step, "total_hits"] = len(shots_hit)
        self._game_step += 1

        # update infos
        next_observations = {
            agent_id: self.get_observation(agent_id) for agent_id in self._agents
        }
        infos["__all__"] = dict()
        return next_observations, rewards, terminated, truncated, infos

    def take_action(self, action, position, direction):
        if action <= 4:
            next_position = self.MOVE_ACTION[action](*position)
            next_direction = direction
        elif action > 4 and action < 8:
            next_direction = self.ROTATE_ACTION[action](direction)
            next_position = position
        else:
            next_position = position
            next_direction = direction
        return next_position, next_direction

    @property
    def is_terminal(
        self,
    ) -> bool:
        if (
            self._game_step > self.max_cycles - 1
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

    def predators_in_line_of_fire(self, position, direction):
        # get the indices of all tiles in the line of fire
        # check if any of the indices are occupied by a predator
        # if yes, then stun the predator
        shots_fired = []
        shots_hit = []

        tiles_affected = []
        if direction == 1:  # UP
            tiles_affected = [
                (position[0] - i, position[1] + j)
                for j in [-1, 0, 1]
                for i in range(1, self.pred_vision + 1)
            ]
        elif direction == 2:  # DOWN
            tiles_affected = [
                (position[0] + i, position[1] + j)
                for j in [-1, 0, 1]
                for i in range(1, self.pred_vision + 1)
            ]
        elif direction == 3:  # RIGHT
            tiles_affected = [
                (position[0] + j, position[1] + i)
                for j in [-1, 0, 1]
                for i in range(1, self.pred_vision + 1)
            ]
        elif direction == 4:  # LEFT
            tiles_affected = [
                (position[0] + j, position[1] - i)
                for j in [-1, 0, 1]
                for i in range(1, self.pred_vision + 1)
            ]

        for agent_id, agent in self._predators.items():
            if not agent.is_alive:
                continue
            if agent.position == position:
                shots_fired.append(agent_id)
                continue
            if agent.position in tiles_affected:
                shots_hit.append(agent_id)
                agent.stun(self.pred_stun_rate)
        return shots_fired, shots_hit

    def get_observation(self, agent_id):
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} is not alive or does not exist.")
        agent = self._predators[agent_id]
        offset_x, offset_y = self.DIRECTION_TO_VECTOR[agent.direction]
        pos_x, pos_y = agent.position
        window_l = 2 * self.pred_vision + 1
        center = window_l // 2
        state = self._game_state.copy()
        for k, v in self._predators.items():
            if not v.is_alive:
                continue
            if v.is_stunned:
                state[v.position[0], v.position[1], self.PREDATOR_CHANNEL] = -1 * (
                    v._stun_count
                )
        observation = state[
            (pos_x - self.pred_vision + offset_x[0]) : (
                pos_x + self.pred_vision + offset_x[1] + 1
            ),
            (pos_y - self.pred_vision + offset_y[0]) : (
                pos_y + self.pred_vision + offset_y[1] + 1
            ),
            :,
        ].copy()
        self_channel = np.zeros((window_l, window_l), np.int32)
        self_channel[center - offset_x[0], center - offset_y[0]] = 1
        return np.dstack([observation, self_channel])

    def __name__(self):
        return "gather_v1"

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

        for agent_id, agent in self._predators.items():
            if not agent.is_alive:
                continue
            if not agent.is_stunned:
                agent_str = f"P{agent_id[-1]}{self.ACTION_TO_STR[agent.direction+4][7]}"
            else:
                agent_str = f"S{agent._stun_count}"
            position = agent.position
            render_array[position[0] - lims, position[1] - lims] = agent_str
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


if __name__ == "__main__":
    config = dict(
        map_size=10,
        max_cycles=100,
        npred=2,
        pred_vision=4,
        nprey=2,
        reward_lone=1.0,
        reward_team=1.0,
        render_mode=None,
        pred_stun_rate=30,
    )

    env = gather_v1(**config)
    # fixed_agent = FixedSwingAgent(env)
    # follower_agent = FollowerAgent(env)
    from discrete_pp_v1 import ChaserAgent

    chaser_agent = ChaserAgent(env)
    agressive_agent = AgressiveAgent(env)
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
            == (window_l, window_l, env.NUM_CHANNELS + 1)
            for agent_id in env.possible_agents
        ]
    )
    assert all(
        [
            env.action_spaces[agent_id].n == env.NUM_ACTIONS
            for agent_id in env.possible_agents
        ]
    )

    stats = []
    for i in range(1000):
        # test reset
        obs, infos = env.reset()
        state = env.state()
        assert isinstance(state, np.ndarray)
        assert state.shape == env._state_space
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)
        # os.system("clear")
        while env.agents:
            state = env.state()
            assert all([agent_id in obs.keys() for agent_id in env._agents])
            pad_width = env.pred_vision
            for k, v in obs.items():
                position = env._predators[k].position
                assert (
                    env._observation_spaces[k].shape == v.shape
                ), f"observation_shape does not match for {k}"
                assert (
                    state[position[0], position[1], env.PREDATOR_CHANNEL] == 1
                ), "Predator not in the correct position"

            assert np.sum(env.state()[:, :, env.PREDATOR_CHANNEL]) == len(
                env._agents
            ), "Predator count should be equal to number of agents"
            assert (
                np.sum(env.state()[:, :, env.PREY_CHANNEL]) == env._nprey
            ), "Prey count should be equal to nprey"
            # actions = {agent_id: env.action_spaces[agent_id].sample() \
            #         for agent_id in env.agents}
            actions = {
                "predator_0": agressive_agent.get_action(obs["predator_0"]),
                "predator_1": chaser_agent.get_action(obs["predator_1"]),
            }

            # print(f"STEP: {env._game_step}")
            # for agent_id, agent in env._predators.items():
            #     print(f"{agent}")
            #     position = agent.position
            #     assert env._game_state[position[0], position[1], 1] == 1, "Predator at incoorect position"

            # print(env.render())
            # print(f"Actions: {[(k, env.ACTION_TO_STR[v]) for k, v in actions.items()]}")
            obs, rewards, terminated, truncated, infos = env.step(actions)
            # print(env.render())
            # # print(f"Rewards: {[(k, v) for k, v in rewards.items()]}")

            # if env._game_history.loc[env._game_step-1, "total_shots"] > 0:
            #     print(f"Shots fired: {env._game_history.loc[env._game_step-1, 'total_shots']}")
            #     print(f"Shots hit: {env._game_history.loc[env._game_step-1, 'total_hits']}")
            # time.sleep(0.5)
            # assert sum of 1s in the predator channel is equals len(env._agents)
            # assert sum of 1s in the prey channel is equals len(env._nprey - env._kill_count)
            # os.system("clear")
            global_state = env.state()
            assert np.sum(global_state[:, :, 2]) == env._nprey

        # observations, sum_rewards, terminated, truncated, infos = env.last()
        total_rewards = env._game_history["total_rewards"].sum()
        # total_rewards += (env._game_history['total_shots'] * -0.1).sum()
        p0_rewards = env._game_history["predator_0_rewards"].sum()
        p1_rewards = env._game_history["predator_1_rewards"].sum()
        if round(total_rewards, 3) != round(p0_rewards + p1_rewards, 3):
            breakpoint()

        stats.append(
            [
                env._game_step,
                env._game_history["total_rewards"].sum(),
                env._game_history["total_shots"].sum(),
                env._game_history["total_hits"].sum(),
                env._game_history["predator_0_rewards"].sum(),
                env._game_history["predator_1_rewards"].sum(),
                env._game_history["predator_0_shots"].sum(),
                env._game_history["predator_1_shots"].sum(),
                env._game_history["predator_0_hits"].sum(),
                env._game_history["predator_1_hits"].sum(),
                env._game_history["predator_0_done"].sum(),
                env._game_history["predator_1_done"].sum(),
            ]
        )

    stats = pd.DataFrame(
        stats,
        columns=[
            "steps",
            "total_rewards",
            "total_shots",
            "total_hits",
            "p0_rewards",
            "p1_rewards",
            "p0_shots",
            "p1_shots",
            "p0_hits",
            "p1_hits",
            "p0_done",
            "p1_done",
        ],
    )
    print(stats.mean())