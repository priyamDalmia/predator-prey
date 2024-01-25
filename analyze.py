from ast import Str
from curses import meta
from logging import config
from math import e, exp, inf
from operator import is_
import os
import re
import time
from typing import Any
from unittest import result
from cycler import L
from matplotlib.pylab import f
import pandas as pd
import numpy as np
from sklearn.utils import resample
from itertools import product

from wandb import agent
from environments.discrete_pp_v1 import ChaserAgent, FollowerAgent, discrete_pp_v1
from environments.discrete_pp_v1 import FixedSwingAgent
from environments.wolfpack_discrete import wolfpack_discrete
from causal_ccm.causal_ccm import ccm
from sklearn.decomposition import PCA
import warnings

import ray
from tests.test_multiagent_2d import CONFIG
import nonlincausality as nlc 
from nonlincausality.nonlincausality import nonlincausalityARIMA, nonlincausalityMLP

from statsmodels.tsa.stattools import grangercausalitytests

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# disable tensorflow cuda 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")
_agent_type_dict = {
    "chaser": ChaserAgent,
    "follower": FollowerAgent,
    "fixed": FixedSwingAgent,
}
POLICY_SETS = ["chaser_follower",  "fixed_follower", "chaser_fixed",
               "chaser_chaser", "follower_chaser", "fixed_chaser"]
INTERVALS = [2, 5]
CONFIG = dict(
    policy_name=None,  # if none specficied, cycle through all in POLICY_SETS
    policy_mapping_fn=None,  # f none specified, will try to infer from policy name
    is_recurrent=False,
    length_fac=100,
    dimensions=["x", "y", "dx", "dy", "PCA_1", "PCA_2"],
    env_config=dict(
        map_size=20,
        npred=2,
        nprey=6,
        pred_vision=6,
        reward_lone = 1.0,
        reward_team = 1.5,
    ),
    ccm_E=5,
    gc_lag=5,
    perform_ccm=True,
    perform_granger_linear=True,
)

CAUSAL_PAIRS = [("predator_0", "predator_1"), ("predator_1", "predator_0")]
CAUSAL_TESTS = ["ccm_1"]  #  "spatial_ccm", "ccm_pval"]
NUM_SAMPLES = 10


# alogrithm policy model container class
class AlgoModel:
    def __init__(self, model):
        self.model = model.eval()

    def compute_single_action(self, obs, explore):
        return self.model.compute_single_action(obs, explore)


# TIME SERIES ANALYSIS
def get_correlation(X, Y):
    return [("correlation", round(np.corrcoef(X, Y)[0][1],4))]

def get_ccm(X, Y, lag):
    E = lag
    tau = 1
    corr, pval = ccm(list(X), list(Y), tau=tau, E=E).causality()
    if pd.isna(corr):
        corr = 0.0
        pval = -1.0
    return [("ccm_score", round(corr, 4)),  ("ccm_pval", round(pval, 4))]


# Returns the results from fitting an OLS model to the data 
def get_granger_linear(X, Y, lag):
    try:
        data = np.array([X, Y]).T.astype(np.float64)
        results = grangercausalitytests(data, maxlag=[lag], verbose=False)
        ssr_ftest = results[lag][0]["ssr_ftest"]
        chi_test = results[lag][0]["ssr_chi2test"]
        return [
        ("F-statistic", ssr_ftest[0]),
        ("F-p", ssr_ftest[1]),
        ("chi2", chi_test[0]),
        ("chi-p", chi_test[1]),
    ]
    except:
        return []


# def get_granger_arima(X, Y, lag):
#     data = np.array([X, Y]).T.astype(np.float64)
#     try:
#         results = nonlincausalityARIMA(x=data, maxlag=[lag], x_test=data, plot=False)
#     except: 
#         return []
#     p_value = results[lag].p_value
#     test_statistic = results[lag]._test_statistic

#     best_errors_X = results[lag].best_errors_X
#     best_errors_XY = results[lag].best_errors_XY
#     cohens_d = np.abs(
#         (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
#         / np.std([best_errors_X, best_errors_XY])
#     )
#     return [
#         ("arima_pval", p_value),
#         ("arima_stat", test_statistic),
#         ("arima_cohen", cohens_d),
#     ]

def get_granger_mlp(X, Y, lag):
    nlen = len(X)
    data = np.array([X, Y]).T.astype(np.float64)
    data_train = data[:int(nlen*0.8)]
    data_test = data[int(nlen*0.8):]
    results = nonlincausalityMLP(
        x=data_train, 
        maxlag=[lag],
        Dense_layers=2,
        Dense_neurons=[100, 100],
        x_test=data_test,
        run=1,
        epochs_num=50,
        batch_size_num=128,
        verbose=False,
        plot=False,
    )

    p_value = results[lag].p_value
    test_statistic = results[lag]._test_statistic

    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY
    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    return [
        ("mlp_pval", p_value),
        ("mlp_stat", test_statistic),
        ("mlp_cohen", cohens_d),
    ]

def play_episode(env, policy_mapping_fn, is_recurrent):
    explore = False
    if explore:
        warnings.warn("Exploration is on!")
    # create a data frame to store the data for
    # agents A and agent B

    step = 0
    obs, info = env.reset()
    center = env.state().shape[0] // 2
    done = False
    if is_recurrent:
        rnn_states = dict()
        last_actions = dict()
        last_rewards = dict()
        for agent_id in env.agents:
            last_actions[agent_id] = 0
            last_rewards[agent_id] = 0
            rnn_states[agent_id] = policy_mapping_fn[agent_id].get_initial_state()
    else:
        rnn_states = None
        last_actions = None
        last_rewards = None

    while env.agents:
        actions = dict()
        for agent_id in env.agents:
            if is_recurrent:
                actions[agent_id], rnn_states[agent_id], _ = policy_mapping_fn[
                    agent_id
                ].compute_single_action(
                    obs[agent_id], state=rnn_states[agent_id], explore=explore,
                    prev_action=last_actions[agent_id], prev_reward=last_rewards[agent_id]
                )
            else:
                actions[agent_id] = policy_mapping_fn[agent_id].compute_single_action(
                    obs[agent_id], explore=explore
                )[0]

        obs, rewards, terminated, truncated, infos = env.step(actions)
        step += 1
        if is_recurrent:
            last_actions = actions
            last_rewards = rewards
    
    return env._game_history[:env._game_step - 1]


def transform_epsiode_history(agents, episode_history):
    col_list = ["x", "y", "action", "dx", "dy", "PCA_1", "PCA_2"]
    selected_data = None  
    for agent_id in agents:
        episode_history[f'{agent_id}_x'] =\
              episode_history[f'{agent_id}_position'].apply(eval).apply(lambda x: x[0]).astype(int)
        episode_history[f'{agent_id}_y'] =\
              episode_history[f'{agent_id}_position'].apply(eval).apply(lambda x: x[1]).astype(int)
        episode_history[f'{agent_id}_dx'] =\
              episode_history[f'{agent_id}_x'].diff(-1).fillna(0)
        episode_history[f'{agent_id}_dy'] =\
              episode_history[f'{agent_id}_y'].diff(-1).fillna(0)
        
        xy_data = episode_history.loc[:, [f"{agent_id}_x", f"{agent_id}_y"]].to_numpy()
        episode_history[f'{agent_id}_PCA_1'] = (
            PCA(n_components=1).fit_transform(xy_data).flatten()
        )

        xy_data = episode_history.loc[:, [f"{agent_id}_x", f"{agent_id}_y", f"{agent_id}_action"]].to_numpy()
        episode_history[f'{agent_id}_PCA_2'] = (
            PCA(n_components=1).fit_transform(xy_data).flatten()
        )
        if selected_data is None:
            selected_data = episode_history.loc[:, [f"{agent_id}_{c}" for c in col_list]]
        else:
            selected_data = pd.concat([
                selected_data,
                episode_history.loc[:, [f"{agent_id}_{c}" for c in col_list]]
            ], axis=1)
    
    selected_data['step'] = selected_data.index
    metadata = dict(
        episode_length = len(selected_data),
        rewards = episode_history['total_rewards'].sum(),
        kills = episode_history['total_kills'].sum(),
        assists = episode_history['total_assists'].sum(),
        predator_0_kills = episode_history['predator_0_kills'].sum(),
        predator_1_kills = episode_history['predator_1_kills'].sum(),
        predator_0_assists = episode_history['predator_0_assists'].sum(),
        predator_1_assists = episode_history['predator_1_assists'].sum(),
        predator_0_rewards = episode_history['predator_0_rewards'].sum(),
        predator_1_rewards = episode_history['predator_1_rewards'].sum(),
    )
    return selected_data, metadata


def analyze_task(
    trial_id,
    env,
    policy_name,
    policy_mapping_fn,
    is_recurrent,
    dimensions,
    length_fac,
    ccm_E,
    gc_lag,
    perform_ccm=False,
    perform_granger_linear=False,
):
    analysis_results = []
    collected_data = pd.DataFrame()
    performance_metrics = []
    num_episodes = 0
    for i in [i * length_fac for i in INTERVALS]:
        # 1. Generate epsiode data
        while len(collected_data) <= int(i):
            episode_data = play_episode(env, policy_mapping_fn, is_recurrent)
            transformed_data, metadata = transform_epsiode_history(env.possible_agents, episode_data)
            performance_metrics.append(metadata)
            transformed_data['episode_num'] = num_episodes
            num_episodes += 1
            if len(collected_data) == 0:
                collected_data = transformed_data
            else:
                collected_data = pd.concat([collected_data, transformed_data])
                collected_data.reset_index(inplace=True, drop=True)
        
        # 1. For each fragment length, do causal analysis
        for pair in CAUSAL_PAIRS:
            for dim in dimensions:
                X: pd.Series = collected_data.loc[:, f"{pair[0]}_{dim}"]
                Y: pd.Series = collected_data.loc[:, f"{pair[1]}_{dim}"]

                results = list()
                results.extend(get_correlation(X, Y))
                if perform_ccm:
                    results.extend(get_ccm(X, Y, ccm_E))
                
                if perform_granger_linear:
                    results.extend(get_granger_linear(Y, X, gc_lag))
                    # results.extend(get_granger_arima(Y, X, gc_lag))

                for result in results:
                    analysis_results.append(
                            [
                                (trial_id, policy_name, *pair, result[0], dim),
                                i,
                                result[1],
                            ]
                        )

    performance_metrics = pd.DataFrame(performance_metrics).mean()
    return analysis_results, performance_metrics


@ray.remote
def remote_analyze_task(*args, **kwargs):
    return analyze_task(*args, **kwargs)


def get_analysis_df(policy_sets: list, dimensions: list, length_fac: int = 100):
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
    traj_length = [i * length_fac for i in INTERVALS]
    analysis_df = pd.DataFrame(index=_df_index, columns=traj_length).fillna(0.0)
    return analysis_df


def perform_causal_analysis(
    num_trials: int, use_ray: bool, analysis_df: pd.DataFrame, env, **kwargs
):
    results = []
    eval_df = None
    if use_ray:
        ray_tasks = [
            remote_analyze_task.remote(i, env, **kwargs) for i in range(num_trials)
        ]
        results = ray.get(ray_tasks)
    else:
        for i in range(num_trials):
            results.append(analyze_task(i, env, **kwargs))
        pass

    for analysis_results, eval_results in results:
        for result in analysis_results:
            analysis_df.loc[result[0], result[1]] = result[2]
        eval_df = (
            eval_results
            if eval_df is None
            else pd.concat([eval_results, eval_df], axis=1)
        )
    return analysis_df, eval_df.T

# @ray.remote
# def print_eval_scores(name, env, policy_mapping_fn, is_recurrent):
#     is_recurrent = False
#     eval_stats = [] 
#     for i in range(500):
#         eval_stats.append(get_episode_record(env, policy_mapping_fn, is_recurrent)[1])
    
#     eval_df = pd.DataFrame(eval_stats)
#     desc_str = f"{name} with env:\n{env.metadata}"
#     return desc_str, pd.concat([eval_df.mean(), eval_df.std()], axis=1)
    
def analyze_fixed_strategies(config, results):
    grouped_dfs = []
    eval_dfs = None

    for policy_set, analyis_df, eval_df in results:
        col_list = list(analyis_df.groupby(['mode', 'agent_a', 'agent_b', 'test', 'dimension']).mean().columns)
        agg_funs = []
        for col in col_list:
            if col != 'run_id':
                agg_funs.append((col, lambda x: round(x.mean(), 2)))
            else:
                agg_funs.append((col, 'count'))
        
        df2 = analyis_df.groupby(['mode', 'agent_a', 'agent_b', 'test', 'dimension']).agg(dict(agg_funs))
        grouped_dfs.append(df2) 
        eval_dfs = pd.concat([
            eval_dfs,
            pd.DataFrame([dict(name=policy_set,**dict(eval_df.mean(1)))])
                ]) if eval_dfs is not None else pd.DataFrame([dict(name=policy_set,**dict(eval_df.mean(1)))])
    
    eval_dfs.set_index('name', inplace=True)
    config = config['env_config']
    filename = f"FA_{config['map_size']}r_{config['reward_lone']}.{config['reward_team']}_s_{config['pred_vision']}.txt"
    with open(f"./results/{filename}", "w") as f:
        f.write(f"""EVALUATION: ENV = {config}\n""")
        f.write(eval_dfs.to_string())
        f.write("\n\n")
        for df in grouped_dfs:
            f.write(df.to_string())
            f.write("\n\n")

if __name__ == "__main__":
    config_dict = CONFIG.copy()
    env = wolfpack_discrete(**config_dict["env_config"])
    anaylze_agents = True 
    evaluate_agents = True
    eval_list = ['chaser_follower', 'fixed_follower', 'chaser_fixed']
    if anaylze_agents:
        results = []
        for policy_name in POLICY_SETS:
            start_time = time.time()
            policy_mapping_fn = {
                f"predator_{i}": _agent_type_dict[agent_class]()
                for i, agent_class in enumerate(policy_name.split("_"))
            }
            analysis_df, eval_df = perform_causal_analysis(
                num_trials=3,
                use_ray=False,
                analysis_df=get_analysis_df(
                    [policy_name], config_dict["dimensions"], config_dict["length_fac"]
                ),
                env=env,
                policy_name=policy_name,
                policy_mapping_fn=policy_mapping_fn,
                is_recurrent=config_dict["is_recurrent"],
                dimensions=config_dict["dimensions"],
                length_fac=config_dict["length_fac"],
                ccm_E=config_dict["ccm_E"],
                gc_lag=config_dict["gc_lag"],
                perform_ccm=config_dict['perform_ccm'],
                perform_granger_linear=config_dict['perform_granger_linear'],
            )
            print(
                f"Time taken for {policy_name}: {(time.time() - start_time)/60:.2f} minutes"
            )
            analysis_df.metadata = env.metadata
            analysis_df.to_csv(f"./results/{policy_name}_analysis.csv")
            eval_df.to_csv(f"./results/{policy_name}_eval.csv")
            results.append((policy_name, analysis_df, eval_df))
        analyze_fixed_strategies(config_dict, results)

    # evaluate_agents = True
    # if evaluate_agents:
    #     ray.init(num_cpus=12)
    #     eval_df = []
    #     ray_tasks = []
    #     for step_penalty in [0.0, 0.01, 0.03]:
    #         for reward_type in ['type_1', 'type_2']:
    #             for map_size in [15, 20, 25]:
    #                 for vision in [2,4,6]:
    #                     env_config = dict(
    #                         map_size=map_size,
    #                         npred=2,
    #                         nprey=6,
    #                         pred_vision=vision,
    #                         reward_type=reward_type,
    #                         prey_type="static",
    #                         step_penalty=step_penalty
    #                     )
    #                     env = discrete_pp_v1(**env_config)
    #                     for policy_name in eval_list:
    #                         policy_mapping_fn = {
    #                             f"predator_{i}": _agent_type_dict[agent_class]()
    #                             for i, agent_class in enumerate(policy_name.split("_"))
    #                         }
    #                         ray_tasks.append(print_eval_scores.remote(f"{policy_name}_{step_penalty}_{reward_type}_{map_size}_{vision}", env, policy_mapping_fn, False))

    #     results = ray.get(ray_tasks)
    #     for result in results:
    #         print(f"""\n\n{result[0]}: \n{result[1]}""")