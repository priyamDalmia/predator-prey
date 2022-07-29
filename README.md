# Predator-Prey Variations

This is the collection of environments inspiried from the classic predator-prey reinforcement learning environment. Each environemnt supports multiple agents and serve as a platform to test multi-agent algorithms.

## Variations

1. Classic Predator-Prey.
2. Communication Predator-Prey.
3. Hunter-Souct Predator-Prey.
4. Reward Sharing Predator-Prey.


## Game features 

1. Explicit Communications - Direct messgaing.
2. Implicit Communications - Reward sharing.

## Alogrithms.

1. Advantage Actor-Critic 
This repo also contains implementations of several variants of the popular multi-agent Actor-Critic Algortihms.
1. Independent Advantage Actor-Critic (A2C) and the Asynchronus version.
2. Independent Soft Actor-Critic with off-policy training.
3. Independent Double-Duelling DQNs with prioritized replay buffer.
4. Centralized Actor-Critc Network. 
5.  


Note: We use Independent to denote that those algorithms are inherently single-agent models in contrast to multi-agent models which 
may require two or more agents (teammates) to function.  

## How the game works?

A variant of the classic Predator-Prey Envrironment with *n* predators (adversaries) and preys. The game ends when all prey have been captured by the predators.

Game Description - 

The game contains added varaints and modes such as:
1. Faster Prey to incoporate harder goals for predators and hence solicit cooperation.
2. Finite (and decaying) hitpoints for predators.
3. Full and Partial Obesrvability modes.
4. Obstacles and custom maps.
5. Scout and hunt mode for the predators where, the scout can move freely and only the hunter can capture prey.
6. Automatic reward sharing (equal or nearest neighbour reward sharing by the environment itself).
7. Parallel Mode (imitating the one step Agent Environment Cycle from [PettingZoo] ()).
8. Respawn Mode for training purposes where predator and prey automatically respawn.

## Getting Started

Clone Repo

Linux : Simply clone the repo, create a virtual env to install the dependecies and enjoy!

```
pip install requirements.txt
```

 ### 1. Training 

 Agent policies are trainered using trainer programs. Three modes for training are provided:
 1. *train_\** - Training single or multiple hetrogenoues policies (predator or prey).
 2. *train_prey_\** - Training single or multiple predator policies while keeping the prey policies static. 
 3. *train_prey_\** - Training single or multiple prey policies while keeping the predator policies static. 

`python train_pred.py` : Trains and saves the best policies.

New trainers must be derived from the  basecalss *Trainer*. See the [trainers]() folder for a more detailed description for configuring training paramters.

 ### 2. Evaluating

 ### 3. Creating new agents

[*random_agent.py*]() contains the simpleset boilerplate code for creating a new agent.  

New agents must be derived from the baseclass *BaseAgent*  See the [agents]() folder for a more detailed description for creating agents.

 ### 4. Replays

 ### 5. Tests


Sample Run: Populates states with characters and print.

```python
python3 test.py 15 -npred 4 -nprey 3
```

Sample game loop

```python
import os
from game import Game 
from common import myparser

args = myparser()
env = Game(args.size, args.npred, args.nprey, 1) 
agents_list = env.agents_list
done = False

while not done:
    actions = []
    for agent in agents_list():
        actions.append(random_action=1)

    env.step(actions)
    done = env.is_done()
    env.render()
env.reset()     

```

## Learning to Share

## Asynchronus Training 

## Phase Potraits

## License
