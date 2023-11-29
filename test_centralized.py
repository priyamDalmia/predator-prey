"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import numpy as np
from gymnasium.spaces import Discrete
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from environments.discrete_pp_v1 import discrete_pp_v1
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


if __name__ == "__main__":
    ray.init(local_mode=True)
    from algorithms.centralized_ppo import TorchCentralizedCriticModel, CentralizedCritic

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )
    env_creator = lambda cfg: ParallelPettingZooEnv(discrete_pp_v1(**cfg))
    register_env('discrete_pp', lambda config: env_creator(config))
    env = env_creator({})
    config = (
        PPOConfig()
        # .experimental(_enable_new_api_stack=False)
        .environment('discrete_pp')
        .framework("torch")
        .rollouts(
            batch_mode="complete_episodes", 
            num_rollout_workers=0,
            create_env_on_local_worker=True,)
        .training(
            model={"custom_model": "cc_model",
                   "conv_filters": [[16, [3, 3], 2]],
                   "fcnet_hiddens": [256, 256],
                   "use_lstm": False,
                   "fcnet_activation": "relu",
                   "conv_activation": "relu",},
            train_batch_size=200,
            _enable_learner_api=False)
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
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(_enable_rl_module_api=False)
    )

    stop = {
        "training_iteration": 5,
    }

    trainer = CentralizedCritic(config=config)
    print(trainer.train())