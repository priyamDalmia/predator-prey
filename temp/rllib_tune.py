from pytest import param
from ray import tune, train 
import ray 
import os 
import time 
import random

NUM_MODELS = 5 

ray.init(num_cpus=1, local_mode=True)
# the function to be trained and tuned 
def train_algo(config):
    # create a algo with 0 rollout workers 
    # train the algo 
    score = 0
    i= 0 
    while True:
        i+=1
        score = config['model_id'] + random.randint(1, 10-i)
        time.sleep(1)
        train.report({'training_iteration': i, 'score': score})
    return dict(score=score)

# trial space 
param_space = {
    'model_id':  tune.grid_search(list(range(NUM_MODELS))),
    'lr': 0.01,}

# resources per trial
tuner = tune.Tuner(
    train_algo, 
    param_space=param_space,
    tune_config=tune.TuneConfig(
        num_samples=2,
        metric="score",
        mode="max",
    ),
    run_config=train.RunConfig(
        stop={"training_iteration": 3},
        verbose=1,
    ),
)
results = tuner.fit()
print(results)

# print(results[0])
# print(results[1])
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.examples.env.simple_corridor import SimpleCorridor

# config = (
#     PPOConfig()
#     .environment(SimpleCorridor, env_config={"corridor_length": 10})
#     # Training rollouts will be collected using just the learner
#     # process, but evaluation will be done in parallel with two
#     # workers. Hence, this run will use 3 CPUs total (1 for the
#     # learner + 2 more for evaluation workers).
#     .rollouts(num_rollout_workers=0)
#     .framework('torch')
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
# )
# stop = {
#     "training_iteration": 3,
# }

# tuner = tune.Tuner(
#     "PPO",
#     param_space=config.to_dict(),
#     run_config=train.RunConfig(stop=stop, verbose=1),
# )
# results = tuner.fit()
# print(results)