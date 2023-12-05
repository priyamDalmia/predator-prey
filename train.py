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
from analyze_ray import maPolicy, analyze
from algorithms.centralized_ppo import TorchCentralizedCriticModel, CentralizedCritic
from ray.rllib.models import ModelCatalog

warnings.filterwarnings("ignore")
env_creator = lambda config: ParallelPettingZooEnv(discrete_pp_v2(**config))

# init the env and the algo
def create_algo(config):
    env = env_creator(config["env_config"])
    if config["algorithm_type"] == "independent":
        algo_config = (
            PPOConfig()
            .framework(framework=config["framework"])
            .callbacks(episodeMetrics)
            .training(
                **config["training"],
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
            .offline_data(output=None)
            .experimental(_enable_new_api_stack=False)
        )
        algo = algo_config.build()
    elif config["algorithm_type"] == "centralized":
        ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)
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
                model={"custom_model": "cc_model",
                   "conv_filters": [[16, [3, 3], 2]],
                   "use_lstm": True,
                   "fcnet_activation": "relu",
                   "conv_activation": "relu",},
                lr = config["training"]["lr"],
                sgd_minibatch_size = config["training"]["sgd_minibatch_size"],
                num_sgd_iter = config["training"]["num_sgd_iter"],
                train_batch_size = config["training"]["train_batch_size"],
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
        )
        algo = CentralizedCritic(config=algo_config)
    elif config["algorithm_type"] == "shared":
        algo_config = (
            PPOConfig()
            .framework(framework=config["framework"])
            .callbacks(episodeMetrics)
            .training(
                **config["training"],
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
            .rl_module(_enable_rl_module_api=True)
            .offline_data(output=None)
        )
        algo = algo_config.build()
    else:
        raise ValueError(f"algorithm_type {config['algorithm_type']} not supported")

    algo.build_config_dict = config
    return algo

class episodeMetrics(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        episode.user_data["assists"] = 0
        episode.user_data["kills"] = 0
        episode.user_data["assists_by_id"] = []
        episode.user_data["kills_by_id"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        episode.user_data["assists"] = episode._last_infos["__common__"]["assists"]
        episode.user_data["kills"] = episode._last_infos["__common__"]["kills"]
        episode.user_data["assists_by_id"] = episode._last_infos["__common__"][
            "assists_by_id"
        ]
        episode.user_data["kills_by_id"] = episode._last_infos["__common__"][
            "kills_by_id"
        ]

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
        episode.custom_metrics["assists"] = episode.user_data["assists"]
        episode.custom_metrics["kills"] = episode.user_data["kills"]
        for agent_id, val in episode.user_data["assists_by_id"].items():
            episode.custom_metrics[f"{agent_id}_assists"] = val
        for agent_id, val in episode.user_data["kills_by_id"].items():
            episode.custom_metrics[f"{agent_id}_kills"] = val

# define the Trainable
def train_algo(config):
    algo = create_algo(config)

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
        if (
            config["wandb"]["wandb_init"]
            and results["training_iteration"] % config["wandb"]["wandb_log_freq"] == 0
        ):
            wandb.log(log_dict)

        if config["stop_fn"](None, results):
            # analysis_df = analyze(algo, config)

            if config["wandb"]["wandb_init"]:
                eval_results = algo.evaluate()['evaluation']
                wandb.summary.update(
                    dict(
                        eval_reward = eval_results['episode_reward_mean'],
                        eval_episode_len = eval_results['episode_len_mean'],
                        eval_assists = eval_results['custom_metrics']['assists_mean'],
                        eval_policy_reward_mean = eval_results['policy_reward_mean'],
                        eval_predator_0_assists = eval_results['custom_metrics']['predator_0_assists_mean'],
                        eval_predator_1_assists = eval_results['custom_metrics']['predator_1_assists_mean'],
                    ))
                # create the two tables and store 
                print(wandb.summary)
                wandb.log(
                    dict(
                        analysis_table=wandb.Table(dataframe=analysis_df.reset_index()),
                        # eval_table=wandb.Table(dataframe=eval_df),
                    )
                )
                wandb.finish()

        train.report(results)

# define the main function
def main():
    # load the yaml
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    register_env(config["env_name"], lambda config: env_creator(config))

    def stop_fn(trial_id, result):
        # Stop training if episode total
        stop = result["episodes_total"] > 25000
        return stop

    config["stop_fn"] = stop_fn
    config["wandb"]["wandb_dir_path"] = str(Path("./wandb").absolute())

    if config["tune"]["tune"]:
        # SET HYPERPARAMETERS for TUNING
        # config["algorithm_type"] = tune.grid_search(["independent", "shared", "centralized"])
        # config["env_config"]["reward_type"] = tune.grid_search(
        #     ["type_1", "type_2", "type_3"]
        # )
        config['model']['use_lstm'] = tune.grid_search([True, False])
    storage_path = str(Path("./experiments").absolute())
    tuner = tune.Tuner(
        tune.with_resources(train_algo, {"cpu": 1}),
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="episode_len_mean",
            mode="min",
            num_samples=config["tune"]["num_samples"],
            max_concurrent_trials=config["tune"]["max_concurrent_trials"],
        ),
        run_config=train.RunConfig(
            verbose=1,
            stop=stop_fn,
            storage_path=storage_path,
            log_to_file=False,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=0,
                checkpoint_at_end=False,
            ),
        ),
    )
    results = tuner.fit()
    for res in results:
        print(res.metrics_dataframe)
    return


if __name__ == "__main__":
    main()
    # # load the yaml fiw we
#     with open("config.yaml", "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     register_env(config["env_name"], lambda config: env_creator(config))

#     # # for config define param space
#     ray.init(num_cpus=6)
#     def stop_fn(trial_id, result):
#         # Stop training if episode total
#         stop = result["episodes_total"] > 500
#         return stop
#     config["stop_fn"] = stop_fn
#     config["wandb"]["wandb_dir_path"] = str(Path("./wandb").absolute())
    
#     # test analyze 
#     # algo = create_algo(config)
#     # print(algo.train())
#     # print(f"EVALUATING {algo} \n\n")
#     # results = algo.evaluate()
#     # analysis_df, eval_df = analyze(algo, config)
#     # print(analysis_df)
#     # breakpoint()
#     # sys.exit()

#    # test tune fit 
#     # config["algorithm_type"] = tune.grid_search(["centralized", "shared", "independent"])
#     config["env_config"]["reward_type"] = tune.grid_search(["type_1", "type_2"])
#     resource_group = tune.PlacementGroupFactory(
#         [{'CPU': 1.0}] + [{'CPU': 1.0}] * 1,
#     )
#     tuner = tune.Tuner(
#         tune.with_resources(train_algo, resource_group),
#         param_space=config,
#         tune_config=tune.TuneConfig(
#             metric="episode_len_mean",
#             mode="min",
#             num_samples=1,
#             max_concurrent_trials=2,
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
