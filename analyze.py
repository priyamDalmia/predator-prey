from ast import Str
from curses import meta
from logging import config
from math import inf
import re
from typing import Any
from matplotlib.pylab import f
import pandas as pd
import numpy as np
from sympy import N
from itertools import product
from environments.discrete_pp_v1 import discrete_pp_v1
from environments.discrete_pp_v1 import FixedSwing
from causal_ccm.causal_ccm import ccm
from sklearn.decomposition import PCA

EMULATION_MODES = ["original", "modified"]
CAUSAL_PAIRS = [("predator_0", "predator_1"), ("predator_1", "predator_0")]
CAUSAL_TESTS = ["ccm", "granger"]#  "spatial_ccm", "ccm_pval"]
DIMENSIONS = ["x", "y"] # "dx", "dy", "reward", "PCA"]
TRAJ_LENGTH = [str(i) for i in range(100, 1001, 100)]
_traj_data = [0 for i in range(100, 1001, 100)]
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


def get_ccm_score(X, Y, tau=4, E=1) -> list[tuple[str, float]]:
    """
    return ccm coorelation and p-value
    """
    corr, pval = ccm(list(X), list(Y), tau=tau, E=E).causality()
    if pd.isna(corr):
        corr = 0.0
        pval = -1.0

    return [("ccm", corr), ("ccm_pval", pval)]


def get_granger_score(X, Y, maxlag=1) -> tuple[str, float]:
    """
    return granger causality and p-value
    """
    import statsmodels as sm
    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.array([X, Y]).T.astype(np.float64)
    results = grangercausalitytests(data, maxlag=5)

    # %% Data generation Y->X
    # np.random.seed(10)
    # y = (
    # np.cos(np.linspace(0, 20, 10_100))
    # + np.sin(np.linspace(0, 3, 10_100))
    # - 0.2 * np.random.random(10_100)
    # )
    # np.random.seed(20)
    # x = 2 * y ** 3 - 5 * y ** 2 + 0.3 * y + 2 - 0.05 * np.random.random(10_100)
    # data = np.vstack([x[:-100], y[100:]]).T
    # #%% Test in case of presence of the causality
    # lags = [50, 150]
    # data_train = data[:7000, :]
    # data_test = data[7000:, :]
    # import copy
    # data_test_measure = copy.copy(data_test)
    # np.random.seed(30)
    # data_test_measure[:1500, 1] = np.random.random(1500)

    # import nonlincausality as nlc
    # results_NN = nlc.nonlincausalityNN(
    #     x=data_train,
    #     maxlag=lags,
    #     NN_config=["l", "dr", "g", "dr", "d", "dr"],
    #     NN_neurons=[5, 0.1, 5, 0.1, 5, 0.1],
    #     x_test=data_test,
    #     run=3,
    #     epochs_num=[50, 100],
    #     learning_rate=[0.001, 0.0001],
    #     batch_size_num=128,
    #     verbose=False,
    #     plot=True,
    # )

    # # ARIMA/ARIMAX models
    # results_ARIMA = nlc.nonlincausalityARIMA(x=data_train, maxlag=lags, x_test=data_train)
    # breakpoint()
    print(results[5][0])
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
                nprey=4,
                prey_type="static",
            )
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
            self.modified_agent = FixedSwing()
            self.mode = "modified"
        else:
            raise ValueError("Invalid mode")

    def compute_actions(self, observations):
        if self.mode == "original":
            actions = {
                agent_id: self.get_action_for_agent(agent_id, obs)
                for agent_id, obs in observations.items()
            } 
        elif self.mode == "modified":
            # modify the get action function of the algorithm
            actions = dict()
            actions["predator_0"] = self.get_action_for_agent(
                "predator_0", observations["predator_0"]
            )
            actions["predator_1"] = self.modified_agent.get_action(
                observations["predator_1"]
            )
            return actions
        else:
            raise ValueError("Set mode before calling compute_actions")

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


# Given a maPolicy object do
def analyze(algo, config):
    analysis_dfs = []
    eval_stats_dfs = []

    for i in range(5):
        analysis, eval_stats = analyze_step(algo, config)
        analysis_dfs.append(analysis.copy())
        eval_stats_dfs.append(eval_stats.mean())
    
    analysis_df = pd.concat(
        analysis_dfs, 
        keys = [f"run_{i}" for i in range(len(analysis_dfs))], 
        axis=0, 
        names=['runs'])
    analysis_df = analysis_df.groupby(level=[1,2,3,4,5]).mean()

    eval_stats_df = pd.DataFrame(eval_stats_dfs)
    eval_stats_df = pd.DataFrame(eval_stats_df.mean())
    return analysis_df, eval_stats_df


def analyze_step(algo, config):
    policy = maPolicy(algo)
    # 1. use maPolicy.env to get the env (RecordWrapper obj)
    env = policy.env
    rollout_env_steps = pd.DataFrame([])
    eval_stats = None
    for mode in EMULATION_MODES:
        policy(mode)
        assert (
            policy.steps_sampled == 0
        ), "policy mode set incorrectly, steps sampled should be 0"

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
                eval_stats = metadata if eval_stats is None else pd.concat([eval_stats, metadata], axis=0).reset_index(drop=True)

                # TODO process metadata - evaluation metrics
            for pair in CAUSAL_PAIRS:
                for dim in DIMENSIONS:
                    X: pd.Series = rollout_env_steps.loc[:, (pair[0], dim)]
                    Y: pd.Series = rollout_env_steps.loc[:, (pair[1], dim)]

                    # CCM analysis
                    print(
                        f"CCM analysis for {pair[0]} and {pair[1]} in {dim} dimension"
                    )
                    if config['analysis']["pref_ccm_analysis"]:
                        results = get_ccm_score(X, Y, tau=1, E=1)
                        for result in results:
                            ANALYSIS_DF.loc[(mode, *pair, result[0], dim), i] = result[
                                1
                            ]

                    # Granger analysis
                    if config['analysis']["pref_granger_analysis"]:
                        results = get_granger_score(X, Y, maxlag=1)

                    # Spatial CCM analysis
                    if config['analysis']["pref_spatial_ccm_analysis"]:
                        raise NotImplementedError("Not implemented yet")

                    # Graph analysis
                    if i == 10000 and config['analysis']["pref_graph_analysis"]:
                        raise NotImplementedError("Not implemented yet")
    return ANALYSIS_DF.copy(), eval_stats


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
            episode_record.loc[step, (agent_id, "PCA")] = 0.0
        step += 1

    for agent_id in env.possible_agents:
        episode_record.loc[:, (agent_id, "dx")] = (
            episode_record.loc[:, (agent_id, "x")].diff().fillna(0)
        )
        episode_record.loc[:, (agent_id, "dy")] = (
            episode_record.loc[:, (agent_id, "y")].diff().fillna(0)
        )
        xy_data = episode_record.loc[:, (agent_id, ["x", "y"])].to_numpy()
        episode_record.loc[:, (agent_id, "PCA")] = (
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
        pref_ccm_analysis=True,
        pref_granger_analysis=True,
        pref_spatial_ccm_analysis=False,
        pref_graph_analysis=False,
    )
    algo = None
    results = analyze(algo, config)
    print(results)
