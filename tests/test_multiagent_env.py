from pettingzoo.test import parallel_api_test
from pettingzoo.test import seed_test
from environments.discrete_pp_v1 import discrete_pp_v1
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

def test_parallel_api():
    config = dict(
        map_size = 20,
        reward_type = "type_1",
        max_cycles = 10000,
        npred = 2,
        pred_vision = 2,
        nprey = 6,
        prey_type = "static",
        render_mode = None
    )
    env = discrete_pp_v1(config)
    env = ParallelPettingZooEnv(env)
    assert isinstance(env, ParallelPettingZooEnv)         


def test_multi_agent_env():
    env_creator = lambda config: ParallelPettingZooEnv(discrete_pp_v1(config))
    # register that way to make the environment under an rllib name
    register_env('discrete_pp', lambda config: env_creator(config))
    return env_creator({})

def test_independent_algo():
    env_creator = lambda config: ParallelPettingZooEnv(discrete_pp_v1(config))  
    register_env('discrete_pp', lambda config: env_creator(config))
    env = env_creator({})
    
    from ray.rllib.algorithms.ppo import PPOConfig
    algo_config = (
        PPOConfig()
        .framework(
            framework="torch")
        .environment(
            "discrete_pp",
            env_config=dict(
                map_size = 20,
                reward_type = "type_1",
                max_cycles = 10000,
                npred = 2,
                pred_vision = 2,
                nprey = 6,
                prey_type = "static",
                render_mode = None
            ))
        .rollouts(
            num_rollout_workers=2,)
        .resources(
            num_gpus=0)
        .multi_agent(
            policies=env.par_env.possible_agents,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    algo = algo_config.build()
    observations, infos = env.par_env.reset() 
    steps = 0
    while env.par_env.agents:
        steps +=1 
        actions = {agent:algo.get_policy(agent).compute_single_action(observations[agent])[0] \
               for agent in env.par_env.agents}
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        msg = f"""step: {steps}\n {print([(k, f"{v:.2f}") for k,v in rewards.items()])}"""
        print(msg)
        break
    env.close()
         