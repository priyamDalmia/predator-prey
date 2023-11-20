import sys
import warnings
import json
import time 
from pathlib import Path
from typing import Dict
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

warnings.filterwarnings("ignore")

# init the env and the algo
def create_algo(config):
    env_creator = lambda cfg: ParallelPettingZooEnv(discrete_pp_v1(**cfg))
    register_env(config['env_name'], lambda config: env_creator(config))
    env = env_creator(config['env_config'])
    algo_config = (
        PPOConfig()
        .framework(framework=config['framework'])
        .training(
            _enable_learner_api=True,
            model={"conv_filters": [[16, [4, 4], 2]]},
            lr = config['training']['lr'], 
            train_batch_size=config['training']['train_batch_size'],
        )
        .environment(
            config['env_name'],
            env_config=config['env_config'],
        )
        .rollouts(
            num_rollout_workers=config['rollouts']['num_rollout_workers'],
            create_env_on_local_worker = True,
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
        .rl_module(_enable_rl_module_api=True)
    )
    algo = algo_config.build()
    return algo


# create wandb name
# create the initial log - wandb run
def init_wandb(config: Dict, algo: Algorithm):
    # create dict from config
    # create wandb name
    algo_config = algo.get_config()
    local_worker_env = algo.workers.local_worker().env.par_env
    env_config = local_worker_env.metadata
    wandb_config = {**config['wandb'], **algo_config, **env_config}
    wandb_name = ''
    wandb_name += f"{config['algorithm_type'][0]}_{config['algorithm_class'][0]}"

    if wandb_config['wandb_init']:
        wandb.init(
            entity=wandb_config['wandb_entity'],
            project=wandb_config['wandb_project'],
            name=wandb_name,
            job_type="train",
            notes=wandb_config['wandb_notes'],
            config=wandb_config,
        )


# train and
# log training steps
def train_algo(config):

    algo = create_algo(config)
    if config['wandb']['wandb_init']:
        response = setup_wandb(
            {'algorithm_type' : config['algorithm_type'],
             'algorithm_class': config['algorithm_class'],
                **(algo.get_config())},
            entity=config['wandb']['wandb_entity'],
            project=config['wandb']['wandb_project'],
            notes=config['wandb']['wandb_notes'],
            rank_zero_only=False,)

    results = {}
    i = 0
    while True:
        results = algo.train()
        
        # log results
        log_dict = dict(
            training_iteration=results["training_iteration"],
            episode_len_mean=results["episode_len_mean"],
            episode_reward_mean=results["episode_reward_mean"],
            num_env_steps_sampled=results["num_env_steps_sampled"],
            num_env_steps_trained=results["num_env_steps_trained"],
            episodes_total=results["episodes_total"],
            time_total_s=results["time_total_s"],
            policy_reward_mean=results["policy_reward_mean"],
        ) 
        if config['wandb']['wandb_init']:
            wandb.log(log_dict)
        train.report(log_dict)



# evaluate and
# return evaluation
def evaluate_algo(algo, config):
    # evaluate for a specific number of env episodes
    results = algo.evaluate()
    if wandb.run is not None:
        # TODO do all calulations here
        wandb.run.summary['epsiode_len_mean'] = results['episode_len_mean']
        wandb.run.summary["evaluation_results"] = json.dumps(results)
    return 

# # arguments 
# import argparse
# parser = argparse.ArgumentParser()

# log final summary (training results and evaluation report) + save models
if __name__ == "__main__":
    # load the yaml fiw we 
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ray.init()
    
    # # for config define param space 
    config['env_config']['map_size'] = tune.grid_search([15, 20])
    config['training']['train_batch_size'] = tune.grid_search([200, 500, 1000])
    config['env_config']['pred_vision'] = tune.grid_search([2, 3]) 
    def stop_fn(trial_id, result):
        # Stop training if episode length is good enough
        stop = result['episodes_total'] > 5000
        if wandb.run is not None and stop:
            wandb.run.summary["training_results"] = result
            # TODO all other calulations here - post training 
        return stop
    tuner = tune.Tuner(
        tune.with_resources(train_algo, {"cpu": 1, "gpu": 0}),
        param_space  = config,
        tune_config=tune.TuneConfig(
            metric="episode_len_mean",
            mode="min",
            num_samples=5,
            max_concurrent_trials=6,
        ),
        run_config=train.RunConfig(
            stop=stop_fn,
        ),
    )
    results = tuner.fit()
    for res in results:
        print(res.metrics_dataframe)

sys.exit()