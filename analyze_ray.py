from ast import Str
from curses import meta
from logging import config
from math import inf
import os
import re
import time
from typing import Any
import skccm
from skccm.utilities import train_test_split
from cycler import L
from matplotlib.pylab import f
import pandas as pd
import numpy as np
from pyparsing import C
from sympy import N
from itertools import product
from environments.discrete_pp_v1 import ChaserAgent, FollowerAgent, discrete_pp_v1
from environments.discrete_pp_v1 import FixedSwingAgent
from causal_ccm.causal_ccm import ccm
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

EMULATION_MODES = ["chaser_follower", "fixed_follower", "chaser_fixed"]
CAUSAL_PAIRS = [("predator_0", "predator_1"), ("predator_1", "predator_0")]
CAUSAL_TESTS = ["ccm_1", "ccm_2"]#  "spatial_ccm", "ccm_pval"]
DIMENSIONS = ["PCA_1", "PCA_2", "PCA_3", "x", "y", "dx", "dy"] # "reward", "PCA"]
LENGTH = 500
NUM_SAMPLES = 10
TRAJ_LENGTH = [str(i) for i in range(LENGTH, (10*LENGTH)+1, LENGTH)]
_traj_data = [0 for i in range(LENGTH, (LENGTH*10)+1, LENGTH)]
_df_data = [
    [mode, *pair, test, dim]
    for mode in EMULATION_MODES
    for pair in CAUSAL_PAIRS
    for test in CAUSAL_TESTS
    for dim in DIMENSIONS
]
_df_index = pd.MultiIndex.from_tuples(
    _df_data, names=["mode", "agent_a", "agent_b", "test", "dimension"]
)
ANALYSIS_DF = pd.DataFrame(index=_df_index, columns=TRAJ_LENGTH).fillna(0.0)
_agent_type_dict = {
    "chaser": ChaserAgent,
    "follower": FollowerAgent,
    "fixed": FixedSwingAgent,
}
ccm_tau = 1
ccm_E = 4

def get_ccm_score(X, Y, tau=ccm_tau, E=ccm_E) -> list[tuple[str, float]]:
    """
    return ccm coorelation and p-value
    """
    corr, pval = ccm(list(X), list(Y), tau=tau, E=E).causality()
    if pd.isna(corr):
        corr = 0.0
        pval = -1.0
    #library lengths to test
    #split the embedded time series
    lag = 1 
    embed = 4
    e1 = skccm.Embed(X)
    e2 = skccm.Embed(Y)
    X1 = e1.embed_vectors_1d(lag,embed)
    X2 = e2.embed_vectors_1d(lag,embed)
    x1tr, _, x2tr, _ = train_test_split(X1, X2, percent=.95)
    _, x1te, _, x2te = train_test_split(X1, X2, percent=.20)

    CCM = skccm.CCM() #initiate the class
    #library lengths to test
    len_tr = len(x1tr)
    lib_lens = np.arange(10, len_tr, 10, dtype='int')
    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=[len(x2te)])
    sc1,sc2 = CCM.score()    
    return [("ccm_1", round(corr, 3)), ("ccm_2", round(sc1[0], 3))] # , ("ccm_pval", round(pval, 2))]


def get_granger_score(X, Y, maxlag=1) -> tuple[str, float]:
    """
    return granger causality and p-value
    """
    import statsmodels as sm
    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.array([X, Y]).T.astype(np.float64)
    results = grangercausalitytests(data, maxlag=5)
    return ("granger", 0.0)


# Given an algorithm do
# 1. Generate a maPolicy object
class maPolicy:
    def __init__(self, algo):
        self.mode = None
        if algo is None:
            self.algorithm = None
            self.algorithm_class = "random"
            self.algorithm_type = "random"
            env_config = dict(
                npred=2,
                nprey=6,
                map_size=15,
                pred_vision=6, 
                prey_type="static",)
            self.env = discrete_pp_v1(**env_config)
        else:
            self.algorithm = algo
            build_config = algo.build_config_dict
            self.algorithm_class = build_config["algorithm_class"]
            self.algorithm_type = build_config["algorithm_type"]
            env_config = algo.config.env_config
            self.env = discrete_pp_v1(**env_config)

    def __call__(self, mode: str) -> Any:
        self.steps_sampled = 0
        if mode == "original":
            self.mode = "original"
            return
        elif mode == "modified":
            # modify the get action function of the algorithm
            self.modified_agent = FixedSwingAgent()
            self.mode = "modified"
        self.mode = mode

    def compute_actions(self, observations):
        actions = dict()
        if self.mode == "original":
            actions = {
                agent_id: self.get_action_for_agent(agent_id, obs)
                for agent_id, obs in observations.items()
            } 
        elif self.mode == "modified":
            # modify the get action function of the algorithm
            actions["predator_0"] = self.get_action_for_agent(
                "predator_0", observations["predator_0"]
            )
            actions["predator_1"] = self.modified_agent.get_action(
                observations["predator_1"]
            )
            return actions
        else:
            for i, agent_type in enumerate(self.mode.split("_")):
                actions[f'predator_{i}'] = _agent_type_dict[agent_type]().get_action(observations[f'predator_{i}'])
            return actions

        self.steps_sampled += 1
        return actions

    def get_action_for_agent(self, agent_id, obs):
        if self.algorithm_type == "random":
            return self.env.action_spaces[agent_id].sample()
        elif self.algorithm_type == "centralized":
            return self.algorithm.get_policy(agent_id)\
                .compute_single_action(obs, explore=False)[0]
        elif self.algorithm_type == "independent":
            return self.algorithm.get_policy(agent_id)\
                .compute_single_action(obs, explore=False)[0]
        elif self.algorithm_type == "shared":
            return self.algorithm.get_policy('shared_policy')\
                .compute_single_action(obs, explore=False)[0]
            # return self.algorithm.get_policy(agent_id)\
                # .compute_single_action(obs, explore=False)[0]
        raise NotImplementedError("Not implemented yet")

import ray 
# Given a maPolicy object do
def analyze(algo, config):
    analysis_dfs = []
    time_start = time.time()    

    for i in range(NUM_SAMPLES):
        analysis = analyze_step(algo, config)
        analysis_dfs.append(analysis.copy())
    
    analysis_df = pd.concat(
        analysis_dfs, 
        keys = [f"run_{i}" for i in range(len(analysis_dfs))], 
        axis=0, 
        names=['runs'])
    # analysis_df = analysis_df.groupby(level=[1,2,3,4,5]).mean()

    # eval_stats_df = pd.DataFrame(eval_stats_dfs)
    # eval_stats_df = pd.DataFrame(eval_stats_df.mean())
    print(f"Time taken: {(time.time() - time_start) / 60:.2f} mins")
    # analysis_df.to_csv("analysis_2.csv")

    analysis_df = analysis_df.groupby(level=[1,2,3,4,5]).mean()
    # analysis_df.to_csv("analysis_2_mean.csv")
    return analysis_df


@ray.remote 
def ray_task(policy, mode, config):
    policy(mode)
    env = policy.env
    rows = []
    rollout_env_steps = pd.DataFrame([])
    # 1. Generate epsiode data
    for i in TRAJ_LENGTH:
        while len(rollout_env_steps) <= int(i):
            episode_record, metadata = get_episode_record(env, policy)
            rollout_env_steps = (
                pd.concat([rollout_env_steps, episode_record], axis=0)
                if len(rollout_env_steps) > 0
                else episode_record
            )
            rollout_env_steps.reset_index(drop=True, inplace=False)
            metadata = pd.DataFrame([metadata])
            # eval_stats = metadata if eval_stats is None else pd.concat([eval_stats, metadata], axis=0).reset_index(drop=True)

            # TODO process metadata - evaluation metrics
        for pair in CAUSAL_PAIRS:
            for dim in DIMENSIONS:
                X: pd.Series = rollout_env_steps.loc[:, (pair[0], dim)]
                Y: pd.Series = rollout_env_steps.loc[:, (pair[1], dim)]

                # CCM analysis
                print(
                    f"{i}-{mode}: CCM analysis for {pair[0]} and {pair[1]} in {dim} dimension; len(X)={len(X)})"
                )
                if config['analysis']["pref_ccm_analysis"]:
                    results = get_ccm_score(X, Y, tau=ccm_tau, E=ccm_E)
                    for result in results:
                        # ANALYSIS_DF.loc[(mode, *pair, result[0], dim), i] = result[
                        #     1
                        # ]
                        rows.append([(mode, *pair, result[0], dim), i, result[1]])

                # Granger analysis
                if config['analysis']["pref_granger_analysis"]:
                    results = get_granger_score(X, Y, maxlag=1)

                # Spatial CCM analysis
                if config['analysis']["pref_spatial_ccm_analysis"]:
                    raise NotImplementedError("Not implemented yet")

                # Graph analysis
                if i == 10000 and config['analysis']["pref_graph_analysis"]:
                    raise NotImplementedError("Not implemented yet")
    return rows 

def analyze_step(algo, config):
    policy = maPolicy(algo)
    analysis_df = ANALYSIS_DF.copy()
    # 1. use maPolicy.env to get the env (RecordWrapper obj)
    env = policy.env
    ray_tasks = [ray_task.remote(policy, mode, config) for mode in EMULATION_MODES]
    results = ray.get(ray_tasks)
    for result in results:
        for row in result:
            analysis_df.loc[row[0], row[1]] = row[2]
    return analysis_df.copy()


def get_episode_record(env, policy):
    # create a data frame to store the data for
    # agents A and agent B
    col_names = [
        ["predator_0", "predator_1"],
        ["x", "y", "dx", "dy", "done", "reward", "action"],
    ]
    col_index = pd.MultiIndex.from_product(col_names, names=["agent_id", "dimension"])
    episode_record = pd.DataFrame([], columns=col_index)
    metadata = dict()
    step = 0
    obs, info = env.reset()
    done = False
    center = env.state().shape[0] // 2
    while env.agents:
        actions = policy.compute_actions(obs)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        for agent_id in env.agents:
            episode_record.loc[step, (agent_id, "x")] = env._predators[
                agent_id
            ].position[0]
            episode_record.loc[step, (agent_id, "y")] = env._predators[
                agent_id
            ].position[1]
            episode_record.loc[step, (agent_id, "dx")] = 0
            episode_record.loc[step, (agent_id, "dy")] = 0
            episode_record.loc[step, (agent_id, "done")] = int(terminated[agent_id])
            episode_record.loc[step, (agent_id, "reward")] = rewards[agent_id]
            episode_record.loc[step, (agent_id, "action")] = actions[agent_id]
            episode_record.loc[step, (agent_id, "PCA_1")] = 0.0
            episode_record.loc[step, (agent_id, "PCA_2")] = 0.0
        # os.system("clear")
        # print(env.render())
        # time.sleep(0.2)
        step += 1
    
    for agent_id in env.possible_agents:
        episode_record.loc[:, (agent_id, "dx")] = (
            episode_record.loc[:, (agent_id, "x")].diff().fillna(0)
        )
        episode_record.loc[:, (agent_id, "dy")] = (
            episode_record.loc[:, (agent_id, "y")].diff().fillna(0)
        )
        
        xy_data = episode_record.loc[:, (agent_id, ["dx", "dy"])].to_numpy()
        episode_record.loc[:, (agent_id, "PCA_1")] = (
            PCA(n_components=1).fit_transform(xy_data).flatten()
        )

        xy_data = episode_record.loc[:, (agent_id, ["dx", "dy", "action"])].to_numpy()
        episode_record.loc[:, (agent_id, "PCA_2")] = (
            PCA(n_components=1).fit_transform(xy_data).flatten()
        )
        
        # noramlize x and y from center
        episode_record.loc[:, (agent_id, "x")] = (
            episode_record.loc[:, (agent_id, "x")].apply(lambda x: (x - center) / (2 * center))
        )

        episode_record.loc[:, (agent_id, "y")] = (
            episode_record.loc[:, (agent_id, "y")].apply(lambda x: (x - center) / (2 * center))
        )
        
        xy_data = episode_record.loc[:, (agent_id, ["x", "y"])].to_numpy()
        episode_record.loc[:, (agent_id, "PCA_3")] = (
            PCA(n_components=1).fit_transform(xy_data).flatten()
        )

    metadata["episode_len"] = step
    metadata["episode_reward"] = sum(env._rewards_sum.values())
    metadata["assists"] = env._assists
    metadata["kills"] = env._kills
    metadata.update(
        {f"{agent_id}_assists": env._assists_by_id[agent_id] for agent_id in env.possible_agents}
    )
    metadata.update(
        {f"{agent_id}_kills": env._kills_by_id[agent_id] for agent_id in env.possible_agents}
    )
    metadata.update(
        {f"{agent_id}_reward": env._rewards_sum[agent_id] for agent_id in env.possible_agents}
    )
    return episode_record, metadata

if __name__ == "__main__":
    config = dict(
        analysis = dict(
            pref_ccm_analysis=True,
            pref_granger_analysis=False,
            pref_spatial_ccm_analysis=False,
            pref_graph_analysis=False,
        )
    )
    ray.init()
    algo = None
    results = analyze(algo, config)
    print(results)
    ray.shutdown()

# %%
