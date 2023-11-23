from re import I
import sys
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

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

# define the callback for wandb and ANALYSIS
class wandbCallback(Callback):
    def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        config = trial.config
        if config['wandb']['wandb_init']:
            response = setup_wandb({
                'algorithm_type' : config['algorithm_type'],
                'algorithm_class': config['algorithm_class'],
                    **(config)},
                entity=config['wandb']['wandb_entity'],
                project=config['wandb']['wandb_project'],
                notes=config['wandb']['wandb_notes'],
                rank_zero_only=False,)
        return super().on_trial_start(iteration, trials, trial, **info)
    
    def on_trial_result(self, iteration: int, trials: List[Trial], trial: Trial, result: Dict, **info):
        log_dict = dict(
            training_iteration=result["training_iteration"],
            episode_len_mean=result["episode_len_mean"],
            episode_reward_mean=result["episode_reward_mean"],
            num_env_steps_sampled=result["num_env_steps_sampled"],
            num_env_steps_trained=result["num_env_steps_trained"],
            episodes_total=result["episodes_total"],
            time_total_s=result["time_total_s"],
            policy_reward_mean=result["policy_reward_mean"],
        ) 
        if wandb.run is not None:
            wandb.log(log_dict)
        return super().on_trial_result(iteration, trials, trial, result, **info)
    
    @classmethod
    def on_trial_completed(cls, algo: Algorithm, result: Dict):
        if wandb.run is not None:
            wandb.run.summary["training_iteration"] = result["training_iteration"]
            wandb.run.summary["episode_len_mean"] = result["episode_len_mean"]
            wandb.run.summary["episode_reward_mean"] = result["episode_reward_mean"]
            wandb.run.summary["num_env_steps_sampled"] = result["num_env_steps_sampled"]
            wandb.run.summary["num_env_steps_trained"] = result["num_env_steps_trained"]
            wandb.run.summary["episodes_total"] = result["episodes_total"]
            wandb.run.summary["time_total_s"] = result["time_total_s"]
            wandb.run.summary["policy_reward_mean"] = result["policy_reward_mean"]
        return 

# define the Trainable
def train_algo(config):
    algo = create_algo(config)
    results = {}
    i = 0
    while True:
        results = algo.train()
        if config['stop_fn'](None, results):
            wandbCallback.on_trial_completed(algo, results)
            break
        train.report(results)

# define the main function
def main():
    # load the yaml 
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    register_env(config['env_name'], lambda config: env_creator(config))
     
    def stop_fn(trial_id, result):
        # Stop training if episode total 
        stop = result['episodes_total'] > 10000
        return stop
    config['stop_fn'] = stop_fn 

    if config['tune']['tune']: 
        # SET HYPERPARAMETERS for TUNING
        config['algorithm_type'] = tune.grid_search(['independent', 'shared'])
        config['training']['model']['conv_filters'] = tune.grid_search([[[16, [2, 2], 2], [16, [4, 4], 1]], [[16, [2, 2], 2]]])
        config['training']['model']['fcnet_hiddens'] = tune.grid_search([[256, 256], [512, 512]])
        config['training']['model']['_disable_preprocessor_api'] = tune.grid_search([True, False])
        config['training']['model']['conv_activation'] = tune.grid_search(['tanh', 'relu'])

    tuner = tune.Tuner(
        train_algo,
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
            log_to_file=True,
            callbacks=[wandbCallback()],
        ),
    )
    results = tuner.fit()
    for res in results:
        print(res.metrics_dataframe)
    return 

if __name__ == "__main__":
    main()

    # # ## DEBUGGING
    # ray.init(
    #     num_cpus=1,
    # )
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
    # # config['env_config']['reward_type'] = tune.grid_search(['type_1', 'type_2'])
    # # config['env_config']['map_size'] = tune.grid_search([15, 20])
    # # config['training']['train_batch_size'] = tune.grid_search([200, 500, 1000])
    # # config['env_config']['pred_vision'] = tune.grid_search([2, 3]) 
    # def stop_fn(trial_id, result):
    #     # Stop training if episode total 
    #     stop = result['episodes_total'] > 50
    #     return stop
    # config['stop_fn'] = stop_fn 

    # tuner = tune.Tuner(
    #     train_algo,
    #     param_space  = config,
    #     tune_config=tune.TuneConfig(
    #         metric="episode_len_mean",
    #         mode="min",
    #         num_samples=5,
    #         max_concurrent_trials=6,
    #     ),
    #     run_config=train.RunConfig(
    #         stop=stop_fn,
    #         storage_path=str(Path('./experiments').absolute()),
    #         log_to_file=True,
    #         callbacks=[wandbCallback()],
    #     ),
    # )
    # results = tuner.fit()
    # for res in results:
    #     print(res.metrics_dataframe)

    # # sys.exit(0)