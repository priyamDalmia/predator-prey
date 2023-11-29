from unittest import result
from pettingzoo.test import parallel_api_test
from pettingzoo.test import seed_test
from sympy import O
from environments.discrete_pp_v1 import discrete_pp_v1
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from gymnasium.spaces import Discrete, Box
from typing import Dict, Tuple
import numpy as np
from ray.rllib.examples.models.centralized_critic_models import YetAnotherTorchCentralizedCriticModel
from ray.rllib.algorithms.ppo import PPOConfig

CONFIG = dict(
    env_name = "discrete_pp",
    framework = "torch",
    env_config = dict(
        map_size = 20,
        reward_type = "type_1",
        max_cycles = 100,
        npred = 2,
        pred_vision = 2,
        nprey = 6,
        prey_type = "static",
        render_mode = None
    ),
    training = dict(
        lr = 0.0001,
        train_batch_size = 256,
    ),
    rollouts = dict(
        num_rollout_workers = 0,
    )
) 

def test_independent_algo():
    from ray.rllib.algorithms.ppo import PPOConfig
    config = CONFIG.copy()
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
    env = algo.workers.local_worker().env.par_env
    observations, infos = env.reset() 
    steps = 0
    while env.agents:
        steps +=1 
        actions = {agent:algo.get_policy(agent).compute_single_action(observations[agent])[0] \
               for agent in env.agents}
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        msg = f"""step: {steps}\n {print([(k, f"{v:.2f}") for k,v in rewards.items()])}"""
        print(msg)
        break
    env.close()

class MyCallbacks(DefaultCallbacks):
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

def test_shared_algo():
    config = CONFIG.copy()
    env_creator = lambda cfg: ParallelPettingZooEnv(discrete_pp_v1(**cfg))
    register_env(config['env_name'], lambda config: env_creator(config))
    env = env_creator(config['env_config'])
    algo_config = (
        PPOConfig()
        .framework(framework=config['framework'])
        .callbacks(MyCallbacks)
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
            num_rollout_workers=0,
            create_env_on_local_worker = True,
        )
        .multi_agent(
            policies={
                "shared_policy"
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),
        )
        .rl_module(_enable_rl_module_api=True)
    )
    algo = algo_config.build()
    
    ## Test single step 
    # env = algo.workers.local_worker().env.par_env
    # # observations, infos = env.reset() 
    # steps = 0
    # while env.agents:
    #     steps +=1 
    #     actions = {agent:algo.get_policy('shared_policy').compute_single_action(observations[agent])[0] \
    #            for agent in env.agents}
    #     # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     msg = f"""step: {steps}\n {print([(k, f"{v:.2f}") for k,v in rewards.items()])}"""
    #     print(msg)
    # #     break
    # env.close()

    ## Test train
    results = algo.train()
    print(results)

from ray.rllib.models import ModelCatalog
class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(2))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, -2:] = opponent_actions


def test_centralized_algo():
    config = CONFIG.copy()
    env_creator = lambda cfg: ParallelPettingZooEnv(discrete_pp_v1(**cfg))
    register_env(config['env_name'], lambda config: env_creator(config))
    env = env_creator(config['env_config'])
    
    ModelCatalog.register_custom_model("cc_model", YetAnotherTorchCentralizedCriticModel)

    action_space = Discrete(env.action_space['predator_0'].n)
    observer_space = dict(
        {
            "own_obs": env.observation_space['predator_0'],
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": env.observation_space['predator_0'],
            "opponent_action": action_space,
        }
    )
    algo_config = (
        PPOConfig()
        .framework(framework=config['framework'])
        .callbacks(FillInActions)
        .training(
            _enable_learner_api=False,
            # model={"conv_filters": [[16, [4, 4], 2]]},
            model={'custom_model': "cc_model"},
            lr = config['training']['lr'], 
            train_batch_size=config['training']['train_batch_size'],
        )
        .environment(
            config['env_name'],
            env_config=config['env_config'],
        )
        .rollouts(
            batch_mode="complete_episodes",
            num_rollout_workers=config['rollouts']['num_rollout_workers'],
            create_env_on_local_worker = True,
            enable_connectors=False,
        )
        .multi_agent(
            policies={
                agent_id: (
                    None,
                    observer_space, # type: ignore
                    action_space, # type: ignore
                    {},
                )   for agent_id in env.par_env.possible_agents
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .rl_module(_enable_rl_module_api=False)
    )
    algo = algo_config.build()