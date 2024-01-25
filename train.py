from email import policy
from ipaddress import _BaseNetwork
from math import log
from re import I
import re
import sys
import os
import warnings
import json
import time
from pathlib import Path
from typing import Dict, List
from numpy import tril
from ray.tune.experiment import Trial
import wandb
import random
import yaml
import ray
from ray import tune, train
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from algorithms.base_model import PPOModel
from analyze import get_analysis_df, perform_causal_analysis
from environments.wolfpack_discrete import wolfpack_discrete
from environments.discrete_pp_v1 import discrete_pp_v1
from environments.discrete_pp_v2 import discrete_pp_v2
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.air.integrations.wandb import setup_wandb
from ray.tune import Callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.policy import Policy
from algorithms.centralized_ppo import TorchCentralizedCriticModel, CentralizedCritic
from ray.rllib.models import ModelCatalog

from analyze import (
    perform_causal_analysis,
    get_analysis_df,
    _agent_type_dict,
    AlgoModel,
)

warnings.filterwarnings("ignore")
env_creator = lambda config: ParallelPettingZooEnv(wolfpack_discrete(**config))


# init the env and the algo
def create_algo(config):
    env = env_creator(config["env_config"])
    if config["algorithm_type"] == "independent":
        ModelCatalog.register_custom_model("cc_model", PPOModel)
        algo_config = (
            PPOConfig()
            .framework(framework=config["framework"])
            .callbacks(episodeMetrics)
            .training(
                model={"custom_model": "cc_model", **config["training"]["model"]},
                lr=config["training"]["lr"],
                use_critic=config["training"]["use_critic"],  # type: ignore
                use_kl_loss=config["training"]["use_kl_loss"],  # type: ignore
                sgd_minibatch_size=config["training"]["sgd_minibatch_size"],  # type: ignore
                num_sgd_iter=config["training"]["num_sgd_iter"],  # type: ignore
                train_batch_size=config["training"]["train_batch_size"],
                _enable_learner_api=False,
            )
            .environment(
                config["env_name"],
                env_config=config["env_config"],
            )
            .rollouts(**config["rollouts"])
            .multi_agent(
                policies={
                    agent_id: (
                        None,
                        env.observation_space[agent_id],  # type: ignore
                        env.action_space[agent_id],  # type: ignore
                        {},
                    )
                    for agent_id in env.par_env.possible_agents
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                policies_to_train=list(env.par_env.possible_agents),
            )
            .rl_module(_enable_rl_module_api=False)
            .reporting(keep_per_episode_custom_metrics=False)
            .offline_data(output=None)
            .evaluation(evaluation_duration=1000)
            .debugging(
                logger_config={
                    # Use the tune.logger.NoopLogger class for no logging.
                    "type": ray.tune.logger.NoopLogger,
                },
            )
        )
        algo = algo_config.build()
        algo.build_config_dict = config
        return algo
    elif config["algorithm_type"] == "shared":
        ModelCatalog.register_custom_model("cc_model", PPOModel)
        algo_config = (
            PPOConfig()
            .framework(framework=config["framework"])
            .callbacks(episodeMetrics)
            .training(
                model={"custom_model": "cc_model", **config["training"]["model"]},
                lr=config["training"]["lr"],
                use_critic=config["training"]["use_critic"],
                use_kl_loss=config["training"]["use_kl_loss"],
                sgd_minibatch_size=config["training"]["sgd_minibatch_size"],
                num_sgd_iter=config["training"]["num_sgd_iter"],
                train_batch_size=config["training"]["train_batch_size"],
                _enable_learner_api=False,
            )
            .environment(
                config["env_name"],
                env_config=config["env_config"],
            )
            .rollouts(**config["rollouts"])
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),
            )
            .rl_module(_enable_rl_module_api=False)
            .reporting(keep_per_episode_custom_metrics=False)
            .offline_data(output=None)
            .evaluation(evaluation_duration=1000)
            .debugging(
                logger_config={
                    # Use the tune.logger.NoopLogger class for no logging.
                    "type": ray.tune.logger.NoopLogger,
                },
            )
        )
        algo = algo_config.build()
        algo.build_config_dict = config
        return algo
    elif config["algorithm_type"] == "centralized":
        ModelCatalog.register_custom_model(
            "centralized_model", TorchCentralizedCriticModel
        )
        algo_config = (
            PPOConfig()
            .environment(
                config["env_name"],
                env_config=config["env_config"],
            )
            .callbacks(episodeMetrics)
            .framework("torch")
            .rollouts(
                **config["rollouts"],
            )
            .training(
                model={
                    "custom_model": "centralized_model",
                    **config["training"]["model"],
                },
                lr=config["training"]["lr"],
                sgd_minibatch_size=config["training"]["sgd_minibatch_size"],
                num_sgd_iter=config["training"]["num_sgd_iter"],
                train_batch_size=config["training"]["train_batch_size"],
                _enable_learner_api=False,
            )
            .multi_agent(
                policies={
                    agent_id: (
                        None,
                        env.observation_space[agent_id],  # type: ignore
                        env.action_space[agent_id],  # type: ignore
                        {},
                    )
                    for agent_id in env.par_env.possible_agents
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                policies_to_train=list(env.par_env.possible_agents),
            )
            .rl_module(_enable_rl_module_api=False)
            .offline_data(output=None)
            .evaluation(evaluation_duration=1000)
            .debugging(
                logger_config={
                    # Use the tune.logger.NoopLogger class for no logging.
                    "type": ray.tune.logger.NoopLogger,
                },
            )
        )
        algo = CentralizedCritic(config=algo_config)
    else:
        raise ValueError(f"algorithm_type {config['algorithm_type']} not supported")

    algo.build_config_dict = config
    return algo


class episodeMetrics(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        env = episode.worker.env.par_env
        episode.custom_metrics["rewards"] = env._game_history["total_rewards"].sum()
        episode.custom_metrics["kills"] = env._game_history["total_kills"].sum()
        episode.custom_metrics["assists"] = env._game_history["total_assists"].sum()
        for agent_id in env.possible_agents:
            episode.custom_metrics[f"{agent_id}_rewards"] = env._game_history[
                f"{agent_id}_rewards"
            ].sum()
            episode.custom_metrics[f"{agent_id}_kills"] = env._game_history[
                f"{agent_id}_kills"
            ].sum()
            episode.custom_metrics[f"{agent_id}_assists"] = env._game_history[
                f"{agent_id}_assists"
            ].sum()

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True


# define the Trainable
def train_algo(config):
    algo = create_algo(config)
    print(f"WORKING DIRECTIRY: {os.getcwd()}")
    print(f"CONFIG: {config}")

    if config["wandb"]["wandb_init"]:
        wandb.init(
            dir=config["wandb"]["wandb_dir_path"],
            config={
                "algorithm_type": config["algorithm_type"],
                "algorithm_class": config["algorithm_class"],
                **(config),
            },
            entity=config["wandb"]["wandb_entity"],
            project=config["wandb"]["wandb_project"],
            name="df" + str(int(time.time()) + random.randint(0, 100000)),
        )

    results = {}
    i = 0
    while True:
        results = algo.train()
        # if config['stop_fn'](None, results):
        #     wandbCallback.on_trial_completed(algo, results)
        #     break

        log_dict = dict(
            training_iteration=results["training_iteration"],
            episode_len_mean=results["episode_len_mean"],
            episode_reward_mean=results["episode_reward_mean"],
            num_env_steps_sampled=results["num_env_steps_sampled"],
            episodes_total=results["episodes_total"],
            time_total_s=results["time_total_s"],
            policy_reward_mean=results["policy_reward_mean"],
            episode_assists_mean=results["custom_metrics"]["assists_mean"],
            episode_kills_mean=results["custom_metrics"]["kills_mean"],
            predator_0_assists=results["custom_metrics"]["predator_0_assists_mean"],
            predator_0_kills=results["custom_metrics"]["predator_0_kills_mean"],
            predator_1_assists=results["custom_metrics"]["predator_1_assists_mean"],
            predator_1_kills=results["custom_metrics"]["predator_1_kills_mean"],
        )

        # if config['wandb']['wandb_init'] and \
        #     (results['training_iteration'] % config['wandb']['wandb_log_freq'] == 0
        #      or results['training_iteration'] == 1):
        if config["wandb"]["wandb_init"] and (
            results["training_iteration"] % config["wandb"]["wandb_log_freq"] == 0
            or results["training_iteration"] == 1
        ):
            wandb.log(log_dict)

        if config["stop_fn"](None, results):
            if config["wandb"]["wandb_init"]:
                wandb.log(log_dict)
                eval_results = algo.evaluate()["evaluation"]
                wandb.summary.update(
                    dict(
                        eval_episodes=eval_results["episodes_this_iter"],
                        eval_reward=eval_results["episode_reward_mean"],
                        eval_episode_len=eval_results["episode_len_mean"],
                        eval_assists=eval_results["custom_metrics"]["assists_mean"],
                        eval_policy_reward_mean=eval_results["policy_reward_mean"],
                        eval_predator_0_assists=eval_results["custom_metrics"][
                            "predator_0_assists_mean"
                        ],
                        eval_predator_1_assists=eval_results["custom_metrics"][
                            "predator_1_assists_mean"
                        ],
                    )
                )
                if config["analysis"]["analysis"]:
                    for policy_i in config["analysis"]["policy_set"]:
                        policy_name = (
                            config["algorithm_type"]
                            if policy_i == "original"
                            else f"{config['algorithm_type']}_{policy_i}"
                        )
                        analysis_df = get_analysis_df(
                            [policy_name],
                            config["analysis"]["dimensions"],
                            config["analysis"]["length_fac"],
                        )

                        # build the policy mapping fn
                        policy_mapping_fn = get_policy_mapping_fn(policy_name, algo)
                        env = env_creator(config["env_config"]).par_env
                        analysis_df, eval_df = perform_causal_analysis(
                            num_trials=config["analysis"]["num_trials"],
                            use_ray=False,
                            analysis_df=analysis_df,
                            env=env,
                            policy_name=policy_name,
                            policy_mapping_fn=policy_mapping_fn,
                            is_recurrent=config["training"]["model"]["use_lstm"],
                            dimensions=config["analysis"]["dimensions"],
                            length_fac=config["analysis"]["length_fac"],
                            ccm_tau=config["analysis"]["ccm_tau"],
                            ccm_E=config["analysis"]["ccm_E"],
                            pref_ccm_analysis=config["analysis"]["pref_ccm_analysis"],
                            pref_granger_analysis=config["analysis"][
                                "pref_granger_analysis"
                            ],
                            pref_spatial_ccm_analysis=config["analysis"][
                                "pref_spatial_ccm_analysis"
                            ],
                            pref_graph_analysis=config["analysis"][
                                "pref_graph_analysis"
                            ],
                        )
                        # create the two tables and store
                        analysis_df.columns = analysis_df.columns.astype(str)
                        analysis_table = wandb.Table(
                            dataframe=analysis_df.reset_index()
                        )
                        eval_table = wandb.Table(dataframe=eval_df)
                        wandb.log(
                            {
                                f"{policy_name}_analysis_df": analysis_table,
                                f"{policy_name}_eval_df": eval_table,
                            }
                        )
                wandb.finish()
                train.report(results)
        if (
            results["training_iteration"] % 20 == 0
            or results["training_iteration"] == 1
        ):
            train.report(results)


# define the main function
def main():
    start = time.time()
    # load the yaml
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    register_env(config["env_name"], lambda config: env_creator(config))
    MAX_EXPISODES = config["tune"]["max_episodes"]

    def stop_fn(trial_id, result):
        # Stop training if episode total
        stop = result["episodes_total"] > MAX_EXPISODES
        return stop

    config["stop_fn"] = stop_fn
    config["wandb"]["wandb_dir_path"] = str(Path("./wandb").absolute())

    if config["tune"]["tune"]:
        # SET HYPERPARAMETERS for TUNING
        config["env_config"]["map_size"] = tune.grid_search([15, 20])
        config["algorithm_type"] = tune.grid_search(
            ["independent", "shared"]
        )
        # config["env_config"]["reward_type"] = tune.grid_search(["type_1", "type_2", "type_3"])

    storage_path = str(Path("./experiments").absolute())
    tuner = tune.Tuner(
        tune.with_resources(train_algo, {"cpu": 0.5}),
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="episode_len_mean",
            mode="min",
            num_samples=config["tune"]["num_samples"],
            max_concurrent_trials=config["tune"]["max_concurrent_trials"],
            trial_name_creator=trail_name_creator,
        ),
        run_config=train.RunConfig(
            verbose=0,
            stop=stop_fn,
            storage_path=storage_path,
            log_to_file=False,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=False,
            ),
        ),
    )
    results = tuner.fit()
    for res in results:
        print(res.metrics_dataframe)

    print(f"Time taken to finish tune: {(time.time() - start)/60:.2f}")
    return


def get_policy_mapping_fn(policy_name, algo):
    policy_maps = dict()
    if len(policy_name.split("_")) > 1:
        train_policy = policy_name.split("_")[0]
        fixed_policy = policy_name.split("_")[1]
        fixed_agent = _agent_type_dict[fixed_policy]()
        if train_policy in ["independent", "centralized"]:
            policy_maps = {
                "predator_0": algo.get_policy("predator_0"),
                "predator_1": fixed_agent,
            }
        elif train_policy == "shared":
            policy_maps = {
                "predator_0": algo.get_policy("shared_policy"),
                "predator_1": fixed_agent,
            }
    else:
        if policy_name in ["independent", "centralized"]:
            policy_maps = {
                "predator_0": algo.get_policy("predator_0"),
                "predator_1": algo.get_policy("predator_1"),
            }
        elif policy_name == "shared":
            policy_maps = {
                "predator_0": algo.get_policy("shared_policy"),
                "predator_1": algo.get_policy("shared_policy"),
            }

    return policy_maps


def trail_name_creator(trail, *args, **kwargs):
    config = trail.config
    name = ""
    name += "r-" if config["training"]["model"]["use_lstm"] else ""
    name = f"{config['algorithm_type'][0]}_"
    name += f"{config['env_config']['reward_type'][-1]}_"
    name += f"{config['env_config']['step_penalty']}_"
    name += f"_{trail.trial_id}"
    return name


if __name__ == "__main__":
    # set global variable
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_SYNCER"] = "1"
    os.environ["TUNE_RESULT_DIR"] = str(Path("./experiments").absolute())
    main()

    # # load the yaml file
    # with open("config.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # register_env(config["env_name"], lambda config: env_creator(config))
    # # create the stop function
    # def stop_fn(trial_id, result):
    #     # Stop training if episode total
    #     stop = result["episodes_total"] > 500
    #     return stop
    # config["stop_fn"] = stop_fn
    # config["wandb"]["wandb_dir_path"] = str(Path("./wandb").absolute())

    # algo = create_algo(config)
    # # # for config define param space
    # ray.init(num_cpus=1)
    # wandb.init(
    #     dir=config["wandb"]["wandb_dir_path"],
    #     config={
    #         "algorithm_type": config["algorithm_type"],
    #         "algorithm_class": config["algorithm_class"],
    #         **(config),
    #     },
    #     entity=config["wandb"]["wandb_entity"],
    #     project=config["wandb"]["wandb_project"],
    # )

    # results = algo.train()
    # print(results)
    # print(f"evaluating {algo} \n\n")
    # results = algo.evaluate()
    # print(results)
    # # create the two tables and store
    # if config["analysis"]["analysis"]:
    #     for policy_i in config["analysis"]["policy_set"]:
    #         policy_name = (
    #             config["algorithm_type"]
    #             if policy_i == "original"
    #             else f"{config['algorithm_type']}_{policy_i}"
    #         )
    #         analysis_df = get_analysis_df(
    #             [policy_name],
    #             config["analysis"]["dimensions"],
    #             config["analysis"]["length_fac"],
    #         )

    #         # build the policy mapping fn
    #         policy_mapping_fn = get_policy_mapping_fn(policy_name, algo)
    #         env = env_creator(config["env_config"]).par_env
    #         analysis_df, eval_df = perform_causal_analysis(
    #             num_trials=config["analysis"]["num_trials"],
    #             use_ray=False,
    #             analysis_df=analysis_df,
    #             env=env,
    #             policy_name=policy_name,
    #             policy_mapping_fn=policy_mapping_fn,
    #             is_recurrent=config["training"]["model"]["use_lstm"],
    #             dimensions=config["analysis"]["dimensions"],
    #             length_fac=config["analysis"]["length_fac"],
    #             ccm_tau=config["analysis"]["ccm_tau"],
    #             ccm_E=config["analysis"]["ccm_E"],
    #             pref_ccm_analysis=config["analysis"]["pref_ccm_analysis"],
    #             pref_granger_analysis=config["analysis"]["pref_granger_analysis"],
    #             pref_spatial_ccm_analysis=config["analysis"][
    #                 "pref_spatial_ccm_analysis"
    #             ],
    #             pref_graph_analysis=config["analysis"]["pref_graph_analysis"],
    #         )
    #         print(analysis_df)
    #         analysis_df.columns = analysis_df.columns.astype(str)
    #         analysis_table = wandb.Table(dataframe=analysis_df.reset_index())
    #         eval_table = wandb.Table(dataframe=eval_df)
    #         wandb.log(
    #             {
    #                 f"{policy_name}_analysis_df" : analysis_table,
    #                 f"{policy_name}_eval_df" : eval_table,
    #             })
    # sys.exit()

    # test tune fit
    # config["algorithm_type"] = tune.grid_search(["centralized", "shared", "independent"])
    # config["env_config"]["reward_type"] = tune.grid_search(["type_1", "type_2"])
#     resource_group = tune.PlacementGroupFactory(
#         [{'CPU': 1.0}] + [{'CPU': 1.0}] * 1,
#     )
#     tuner = tune.Tuner(
#         tune.with_resources(train_algo, {"cpu": 1}),
#         param_space=config,
#         tune_config=tune.TuneConfig(
#             metric="episode_len_mean",
#             mode="min",
#             num_samples=1,
#             max_concurrent_trials=2,
#             trial_name_creator=trail_name_creator,
#         ),
#         run_config=train.RunConfig(
#             stop=stop_fn,
#             storage_path=str(Path("./experiments").absolute()),
#             checkpoint_config=train.CheckpointConfig(
#                 checkpoint_frequency=0,
#                 checkpoint_at_end=False,
#             ),
#         ),
#     )
#     results = tuner.fit()
#     for res in results:
#         print(res.metrics_dataframe)

#     sys.exit(0)
# #
