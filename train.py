from math import log
from re import I
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
import yaml 
import ray
from ray import tune, train
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from environments.discrete_pp_v1 import discrete_pp_v1
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.air.integrations.wandb import setup_wandb
from ray.tune import Callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.policy import Policy
warnings.filterwarnings("ignore")
env_creator = lambda config:\
    ParallelPettingZooEnv(discrete_pp_v1(**config))

# init the env and the algo
def create_algo(config):
    env = env_creator(config['env_config'])
    if config['algorithm_type'] == 'independent':
        algo_config = (
            PPOConfig()
            .framework(framework=config['framework'])
            .callbacks(episodeMetrics)
            .training(
                **config['training'],
            )
            .environment(
                config['env_name'],
                env_config=config['env_config'],
            )
            .rollouts(
                **config['rollouts']
            )
            .multi_agent(
                policies={
                    agent_id: (
                        None,
                        env.observation_space[agent_id], # type: ignore
                        env.action_space[agent_id], # type: ignore
                        {},
                    )
                    for agent_id in env.par_env.possible_agents
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                policies_to_train=list(env.par_env.possible_agents),
            )
            .rl_module(_enable_rl_module_api=True))
    elif config['algorithm_type'] == 'centralized':
        raise NotImplementedError
    elif config['algorithm_type'] == 'shared':
        algo_config = (
            PPOConfig()
            .framework(framework=config['framework'])
            .callbacks(episodeMetrics)
            .training(
                **config['training'],
            )
            .environment(
                config['env_name'],
                env_config=config['env_config'],
            )
            .rollouts(
                **config['rollouts']
            )
            .multi_agent(
                policies={
                        "shared_policy"
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),
            )
            .rl_module(_enable_rl_module_api=True))
    else:
        raise ValueError(f"algorithm_type {config['algorithm_type']} not supported")

    algo = algo_config.build()
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
        episode.user_data['assists'] = episode._last_infos['__common__']['assists']
        episode.user_data['kills'] = episode._last_infos['__common__']['kills']
        episode.user_data['assists_by_id'] = episode._last_infos['__common__']['assists_by_id']
        episode.user_data['kills_by_id'] = episode._last_infos['__common__']['kills_by_id']

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
        episode.custom_metrics["assists"] = episode.user_data['assists'] 
        episode.custom_metrics["kills"] = episode.user_data['kills'] 
        for agent_id, val in episode.user_data['assists_by_id'].items():
            episode.custom_metrics[f'{agent_id}_assists'] = val
        for agent_id, val in episode.user_data['kills_by_id'].items():
            episode.custom_metrics[f'{agent_id}_kills'] = val

# define the Trainable
def train_algo(config):
    algo = create_algo(config)
    import random
    if config['wandb']['wandb_init']:
        wandb.init(
            dir = config['wandb']['wandb_dir_path'],
            config = {
                'algorithm_type' : config['algorithm_type'],
                'algorithm_class': config['algorithm_class'],
                    **(config)},
            entity=config['wandb']['wandb_entity'],
            project=config['wandb']['wandb_project'],
            name = "df" + str(int(time.time())+random.randint(0, 100000)),
            mode = "offline"
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
            assists = results['custom_metrics']['assists_mean'],
            kills = results['custom_metrics']['kills_mean'],
            predator_0_assists = results['custom_metrics']['predator_0_assists_mean'],
            predator_0_kills = results['custom_metrics']['predator_0_kills_mean'],
            predator_1_assists = results['custom_metrics']['predator_1_assists_mean'],
            predator_1_kills = results['custom_metrics']['predator_1_kills_mean'],
        ) 

        # if config['wandb']['wandb_init'] and \
        #     (results['training_iteration'] % config['wandb']['wandb_log_freq'] == 0
        #      or results['training_iteration'] == 1):
        wandb.log(log_dict)   
        
        if config['stop_fn'](None, results):
            print("STOPPING TRAINING")
            wandb.finish()
        
        train.report(results)

# define the main function
def main():
    # load the yaml 
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    register_env(config['env_name'], lambda config: env_creator(config))
     
    def stop_fn(trial_id, result):
        # Stop training if episode total 
        stop = result['episodes_total'] > 25000
        return stop
    config['stop_fn'] = stop_fn 
    config['wandb']['wandb_dir_path'] = str(Path('./wandb').absolute())

    if config['tune']['tune']: 
        # SET HYPERPARAMETERS for TUNING
        config['algorithm_type'] = tune.grid_search(['independent', 'shared'])
        config['env_config']['reward_type'] = tune.grid_search(['type_1', 'type_2', 'type_3'])

    tuner = tune.Tuner(
        tune.with_resources(train_algo, {'cpu': 1}),
        param_space  = config,
        tune_config=tune.TuneConfig(
            metric="episode_len_mean",
            mode="min",
            num_samples=config['tune']['num_samples'],
            max_concurrent_trials=config['tune']['max_concurrent_trials'],
        ),
        run_config=train.RunConfig(
            stop=stop_fn,
            storage_path=str(Path('./experiments').absolute()),
            log_to_file=False,
        ),
    )
    results = tuner.fit()
    for res in results:
        print(res.metrics_dataframe)
    return 

if __name__ == "__main__":
    main()

    # # load the yaml fiw we 
    # with open("config.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # register_env(config['env_name'], lambda config: env_creator(config))
    
    # # # for config define param space 
    # # algo = create_algo(config)
    # # algo.train()
    # # results = algo.evaluate()
    # # print(results)
    # config['algorithm_type'] = tune.grid_search(['independent', 'shared'])
    # config['env_config']['reward_type'] = tune.grid_search(['type_1', 'type_2'])
    # # config['env_config']['map_size'] = tune.grid_search([15, 20])
    # # config['training']['train_batch_size'] = tune.grid_search([200, 500, 1000])
    # # config['env_config']['pred_vision'] = tune.grid_search([2, 3]) 
    # def stop_fn(trial_id, result):
    #     # Stop training if episode total 
    #     stop = result['episodes_total'] > 200
    #     return stop
    # config['stop_fn'] = stop_fn 
    # config['wandb']['wandb_dir_path'] = str(Path('./wandb').absolute())

    # tuner = tune.Tuner(
    #     tune.with_resources(train_algo, {'cpu': 1}),
    #     param_space  = config,
    #     tune_config=tune.TuneConfig(
    #         metric="episode_len_mean",
    #         mode="min",
    #         num_samples=1,
    #         max_concurrent_trials=4,
    #     ),
    #     run_config=train.RunConfig(
    #         stop=stop_fn,
    #         storage_path=str(Path('./experiments').absolute()),
    #     ),
    # )
    # results = tuner.fit()
    # for res in results:
    #     print(res.metrics_dataframe)

    # # sys.exit(0)