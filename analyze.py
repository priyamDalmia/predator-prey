from ast import Str
from curses import meta
from logging import config
from math import e, inf
from operator import is_
import os
import re
import time
from typing import Any
from cycler import L
from matplotlib.pylab import f
import pandas as pd
import numpy as np
from pyparsing import C
from sympy import N, O, per
from itertools import product

from wandb import agent
from analyze_ray import analyze
from environments.discrete_pp_v1 import ChaserAgent, FollowerAgent, discrete_pp_v1
from environments.discrete_pp_v1 import FixedSwingAgent
from causal_ccm.causal_ccm import ccm
from sklearn.decomposition import PCA
import warnings

import ray
from tests.test_multiagent_2d import CONFIG
warnings.filterwarnings("ignore")
_agent_type_dict = {
    "chaser": ChaserAgent,
    "follower": FollowerAgent,
    "fixed": FixedSwingAgent,
}
POLICY_SETS = ["chaser_follower", "fixed_follower", "chaser_fixed"]
CONFIG = dict(
    policy_name=None, # if none specficied, cycle through all in POLICY_SETS
    policy_mapping_fn=None, # f none specified, will try to infer from policy name 
    is_recurrent=False,
    length_fac=100,
    dimensions = ["x", "y", "dx", "dy", "PCA_1", "PCA_2", "PCA_3"],
    env_config = dict(
        map_size = 15,
        npred =2 ,
        nprey = 6,
        pred_vision=6,
        reward_type="type_1",
        prey_type="static",),
    ccm_tau = 1,
    ccm_E = 4,
    pref_ccm_analysis=True,
    pref_granger_analysis=False,
    pref_spatial_ccm_analysis=False,
    pref_graph_analysis=False, 
)

CAUSAL_PAIRS = [("predator_0", "predator_1"), ("predator_1", "predator_0")]
CAUSAL_TESTS = ["ccm_1"]#  "spatial_ccm", "ccm_pval"]
NUM_SAMPLES = 10

def get_ccm_score(X, Y, tau, E) -> list[tuple[str, float]]:
    """
    return ccm coorelation and p-value
    """
    corr, pval = ccm(list(X), list(Y), tau=tau, E=E).causality()
    if pd.isna(corr):
        corr = 0.0
        pval = -1.0
    return [("ccm_1", round(corr, 3))] # , ("ccm_pval", round(pval, 2))]

def get_granger_score(X, Y, maxlag=1) -> tuple[str, float]:
    """
    return granger causality and p-value
    """
    import statsmodels as sm
    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.array([X, Y]).T.astype(np.float64)
    results = grangercausalitytests(data, maxlag=5)
    return [("granger", 0.0)]

def get_episode_record(env, policy_mapping_fn, is_recurrent):
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
    center = env.state().shape[0] // 2
    done = False
    while env.agents:
        actions = dict()
        for agent_id in env.agents:
            if is_recurrent:
                raise NotImplementedError("Not implemented yet")
            else:
                actions[agent_id] = policy_mapping_fn[agent_id].compute_single_action(obs[agent_id], explore=False)[0]

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

def analyze_task(
    trial_id,
    env,
    policy_name,
    policy_mapping_fn,
    is_recurrent,
    dimensions,
    length_fac,
    ccm_tau,
    ccm_E,
    pref_ccm_analysis=False,
    pref_granger_analysis=False,
    pref_spatial_ccm_analysis=False,
    pref_graph_analysis=False,
):
    analysis_results = []
    rolled_out_env_steps = pd.DataFrame([])
    eval_stats = None
    for i in range(length_fac, (10*length_fac)+1, length_fac):
        # 1. Generate epsiode data 
        while len(rolled_out_env_steps) <= int(i):
            episode_record, metadata = get_episode_record(env, policy_mapping_fn, is_recurrent)
            rolled_out_env_steps = (
                pd.concat([rolled_out_env_steps, episode_record], axis=0)
                if len(rolled_out_env_steps) > 0
                else episode_record
            )
            rolled_out_env_steps.reset_index(drop=True, inplace=False)
            metadata = pd.DataFrame([metadata])
            eval_stats = metadata if eval_stats is None else pd.concat([eval_stats, metadata], axis=0).reset_index(drop=True)
        # 2. For each fragment length, do causal analysis
        for pair in CAUSAL_PAIRS:
            for dim in dimensions:
                X: pd.Series = rolled_out_env_steps.loc[:, (pair[0], dim)]
                Y: pd.Series = rolled_out_env_steps.loc[:, (pair[1], dim)]

                # CCM analysis
                print(
                    f"{i}-{policy_name}: CCM analysis for {pair[0]} and {pair[1]} in {dim} dimension; len(X)={len(X)})"
                )
                if pref_ccm_analysis:
                    results = get_ccm_score(X, Y, tau=ccm_tau, E=ccm_E)
                    for result in results:
                        analysis_results.append(
                            [(trial_id, policy_name, *pair, result[0], dim), i, result[1]]
                        )

                # Granger analysis
                if pref_granger_analysis:
                    results = get_granger_score(X, Y, maxlag=1)
                    for result in results:
                        analysis_results.append(
                            [(trial_id, policy_name, *pair, result[0], dim), i, result[1]]
                        )
                # Spatial CCM analysis
                if pref_spatial_ccm_analysis:
                    raise NotImplementedError("Not implemented yet")

                # Graph analysis
                if i == 10000 and pref_graph_analysis:
                    raise NotImplementedError("Not implemented yet")
    
    eval_stats_df = pd.DataFrame(eval_stats).mean()
    return analysis_results, eval_stats_df

@ray.remote
def remote_analyze_task(*args, **kwargs):
    return analyze_task(*args, **kwargs)

def get_analysis_df(
        policy_sets: list,
        dimensions: list,
        length_fac: int = 100
):
    _df_data = [
        [0, policy, *pair, test, dim]
        for policy in policy_sets
        for pair in CAUSAL_PAIRS
        for test in CAUSAL_TESTS
        for dim in dimensions
    ]
    _df_index = pd.MultiIndex.from_tuples(
        _df_data, names=["run_id", "mode", "agent_a", "agent_b", "test", "dimension"]
    )
    traj_length = [i for i in range(length_fac, (10*length_fac)+1, length_fac)]
    analysis_df = pd.DataFrame(index=_df_index, columns=traj_length).fillna(0.0)
    return analysis_df

def perform_causal_analysis(
        num_trials: int,
        use_ray: bool,
        analysis_df: pd.DataFrame,
        env,
        **kwargs
    ):

    results = []
    eval_scores = None
    if use_ray:
        ray_tasks = [remote_analyze_task.remote(
            i,
            env,
            **kwargs
        ) for i in range(num_trials)]
        results = ray.get(ray_tasks)
    else:
        for i in range(num_trials):
            results.append(analyze_task(
                i,
                env,
                **kwargs
            ))
        pass

    for analysis_results, eval_df in results:
        for result in analysis_results:
            analysis_df.loc[result[0], result[1]] = result[2]
        eval_scores = eval_df if eval_scores is None else pd.concat([eval_scores, eval_df], axis=0)
    if isinstance(eval_scores, pd.DataFrame):
        eval_df = eval_scores.mean()
    else:
        eval_df = eval_scores
    return analysis_df, eval_df

if __name__ == "__main__":
    config_dict = CONFIG.copy()
    env = discrete_pp_v1(**config_dict["env_config"])
    for policy_name in POLICY_SETS:
        start_time = time.time()
        policy_mapping_fn = {
            f'predator_{i}': _agent_type_dict[agent_class]() for i, agent_class in enumerate(policy_name.split("_"))}
        analysis_df, eval_df = perform_causal_analysis(
            num_trials = 2,
            use_ray = True,
            analysis_df = get_analysis_df(
                [policy_name],
                config_dict["dimensions"],
                config_dict["length_fac"]
            ),
            env = env,
            policy_name=policy_name,
            policy_mapping_fn=policy_mapping_fn,
            is_recurrent=config_dict["is_recurrent"],
            dimensions=config_dict["dimensions"],
            length_fac=config_dict["length_fac"],
            ccm_tau=config_dict["ccm_tau"],
            ccm_E=config_dict["ccm_E"],
            pref_ccm_analysis=config_dict["pref_ccm_analysis"],
            pref_granger_analysis=config_dict["pref_granger_analysis"],
            pref_spatial_ccm_analysis=config_dict["pref_spatial_ccm_analysis"],
            pref_graph_analysis=config_dict["pref_graph_analysis"],
        )
        print(f"Time taken for {policy_name}: {(time.time() - start_time)/60:.2f} minutes")
        analysis_df.to_csv(f"./experiments/{policy_name}_analysis.csv")
