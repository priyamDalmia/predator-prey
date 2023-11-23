from ast import Str
from curses import meta
from logging import config
from math import inf
import re
from typing import Any
import pandas as pd
from sympy import N 
from itertools import product
from environments.discrete_pp_v1 import discrete_pp_v1

EMULATION_MODES = ["original", "modified"]
CAUSAL_PAIRS = [("predator_0", "predator_1"), ("predator_1", "predator_0")]
CAUSAL_TESTS = ["ccm", "granger", "spatial_ccm", "ccm_pval"]
DIMENSIONS = ["x", "y", "dx", "dy", "action", "reward"]
TRAJ_LENGTH = [str(i) for i in range(1000, 10001, 1000)]
_traj_data = [0 for i in range(1000, 10001, 1000)]
_df_data = [
    [mode, *pair, test, dim]
    for mode in EMULATION_MODES
    for pair in CAUSAL_PAIRS
    for test in CAUSAL_TESTS
    for dim in DIMENSIONS
]
_df_index = pd.MultiIndex.from_tuples(
    _df_data,
    names=["mode", "agent_a", "agent_b", "test", "dimension"]
)
ANALYSIS_DF = pd.DataFrame(index=_df_index, columns=TRAJ_LENGTH).fillna(0.0)

def get_ccm_score(X, Y, tau=1, E=1) -> list[tuple[str, float]]:
    """
    return ccm coorelation and p-value
    """
    return [("ccm", 0.0), ("ccm_pval", 0.0)]

def get_granger_score(X, Y, maxlag=1) -> tuple[str, float]:
    """
    return granger causality and p-value
    """
    return ("granger", 0.0)

# Given an algorithm do
# 1. Generate a maPolicy object
class maPolicy:
    def __init__(self, algo):
        if algo is None:
            self.algorithm_class = "random"
            self.algorith_type = "random"
            env_config = dict(
                npred = 2,
                nprey = 4,
                prey_type = "static",
            )
            self.env = discrete_pp_v1(**env_config)
            return
        raise NotImplementedError("Not implemented yet")
    
    def __call__(self, mode: str) -> Any:
        self.steps_sampled = 0
        if mode == 'original':
            return
        elif mode == 'modified':
            # modify the get action function of the algorithm
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid mode")
    
    def compute_actions(self, observations):
        actions = {agent_id: self.get_action_for_agent(agent_id, obs)
                   for agent_id, obs in observations.items()}
        self.steps_sampled += 1
        return actions

    def get_action_for_agent(self, agent_id, obs):
        if self.algorith_type == "random":
            return self.env.action_spaces[agent_id].sample()
        else:
            raise NotImplementedError("Not implemented yet")

# Given a maPolicy object do
def analyze(maPolicy, config):
    # 1. use maPolicy.env to get the env (RecordWrapper obj)
    env = maPolicy.env
    rollout_env_steps = []
    
    for mode in EMULATION_MODES:
        maPolicy(mode)
        assert maPolicy.steps_sampled == 0, "policy mode set incorrectly, steps sampled should be 0"

        # 1. Generate epsiode data
        for i in TRAJ_LENGTH:
            episode_record, metadata = get_episode_record(env, maPolicy)
            rollout_env_steps.append(episode_record)
            for pair in CAUSAL_PAIRS:
                for dim in DIMENSIONS:
                    X: pd.Series = episode_record.loc[:, (pair[0], dim)]
                    Y: pd.Series = episode_record.loc[:, (pair[1], dim)]

                    # CCM analysis
                    if config['pref_ccm_analysis']:
                        results = get_ccm_score(X, Y, tau=1, E=1)
                        for result in results:
                            ANALYSIS_DF.loc[(mode, *pair, result[0], dim), i] = result[1]
                    
                    # Granger analysis
                    if config['pref_granger_analysis']:
                        raise NotImplementedError("Not implemented yet")
                    
                    # Spatial CCM analysis
                    if config['pref_spatial_ccm_analysis']:
                        raise NotImplementedError("Not implemented yet")
                    
                    # Graph analysis
                    if i==10000 and config['pref_graph_analysis']:
                        raise NotImplementedError("Not implemented yet")

def get_episode_record(env, policy):
    # create a data frame to store the data for
    # agents A and agent B
    col_names = [
        ['predator_0', 'predator_1'],
        ['x', 'y', 'dx', 'dy', 'done', 'reward', 'action']
    ]
    col_index = pd.MultiIndex.from_product(col_names, names=["agent_id", "dimension"])
    episode_record = pd.DataFrame([], columns=col_index)
    metadata = dict()
    step = 0
    obs, info = env.reset()
    done = False
    while env.agents:
        actions = policy.compute_actions(obs)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        for agent_id in env.agents:
            episode_record.loc[step, (agent_id, 'x')] = env._predators[agent_id].position[0]
            episode_record.loc[step, (agent_id, 'y')] = env._predators[agent_id].position[1]
            episode_record.loc[step, (agent_id, 'dx')] = 0
            episode_record.loc[step, (agent_id, 'dy')] = 0
            episode_record.loc[step, (agent_id, 'done')] = int(terminated[agent_id])
            episode_record.loc[step, (agent_id, 'reward')] = rewards[agent_id]
            episode_record.loc[step, (agent_id, 'action')] = actions[agent_id]
        step += 1
    metadata['steps'] = step
    metadata['assists'] = env._assists
    metadata['kills'] = env._kills
    return episode_record, metadata

if __name__ == "__main__":
    config = dict(
        pref_ccm_analysis = False,
        pref_granger_analysis = False,
        pref_spatial_ccm_analysis = False,
        pref_graph_analysis = False,
    )
    algo = None 
    maPolicy = maPolicy(algo)
    analyze(maPolicy, config)
