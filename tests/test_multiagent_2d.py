from unittest import result
from environments.discrete_pp_v2 import discrete_pp_v2
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
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

config = CONFIG.copy()
env_creator = lambda cfg: ParallelPettingZooEnv(discrete_pp_v2(**cfg))
register_env(config['env_name'], lambda config: env_creator(config))
env = env_creator(config['env_config'])
algo_config = (
    PPOConfig()
    .framework(framework=config['framework'])
    .training(
        _enable_learner_api=False,
        model = {
            "use_lstm": True,
        },
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
    .rl_module(_enable_rl_module_api=False)
)

def test_independent_algo_with_env():
    algo = algo_config.build()
    env = algo.workers.local_worker().env.par_env
    observations, infos = env.reset() 
    steps = 0
    rnn_states = {agent: algo.get_policy(agent).get_initial_state() for i, agent in enumerate(env.agents)}
    while env.agents:

        steps +=1 
        actions = {}
        for agent in env.agents:
            action, rnn_state, _ = algo.get_policy(agent).compute_single_action(
                observations[agent],
                rnn_states[agent]
            )
            actions[agent] = action
            rnn_states[agent] = rnn_state

        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        msg = f"""step: {steps}\n {print([(k, f"{v:.2f}") for k,v in rewards.items()])}"""
        print(msg)
        break
    env.close()
    return True


def test_independent_algo_with_train_and_evaluation():
    algo = algo_config.build()
    # test algo train 
    results = algo.train()
    print(results)

    # test algo evaluation
    results = algo.evaluate()
    print(results)
    return True
