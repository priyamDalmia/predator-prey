### Towards Interpretable Multi-Agent Reinforcement Learning through Visualizing Learning Dynamics.

An analysis of the evolving State-Values functions using vector fields, of a Multi-Agent System under in a partially-observable, stochastic and non-stationary environment.
In this study, we propose the use of visualizing the state-values of a multi-agent systems under differnet contrainsts to help a more thorough understanding of the agent polices and under-the-hood decision making.


#### Notes on the Experiments, the configuration and results.

All the experiments are carried on a 30X30 grid with an observation window of size 9.

1. Three Predators and N Prey.
** Fair Game vs Advantage to Prey.
** Independent Learning.
** Reward Structure - Individuals.
** Algorithms Implemented:
   2. A2C.

2. Two Predator and Five Prey. 
** Fair Game.
** Independent Learning. 
** Individual Rewards.
** Alogrithms Implemented:
   1. Duel Double DQN
   2. Soft-Adv Actor Critic.

3. Two Predators and Five Prey. 
** Fair Game.
** Independent Learning.
** Combined Team Rewards.
** Algorithms Implemented:
   1. Soft-Adv Actor Critic.
    

4. Two Predators and Five Prey.
** Advantage to Prey. (First Moves + Skip Turns)
** Independent Learning. 
** Combined Team Rewards.
** Algorithms Implemented:
   1. Soft-Adv Actor Critic.

4. Two Predators and Five Prey.
** Advantage to Prey.
** Centralized Learning.
** Combined Team Rewards.
** Algorithms Implemented:
   1. Soft-Adv Actor Critic.
   2. Centralized Counter-Factual Critic.

5. Three Predators and Six Prey.
** Advantage to Prey.
** Decentralized Learning.
** Combined Team Rewards.
** Algorithms Implemented:
   1. Centralized Counter-Factual Critic.
   2. (Learning to Share)

6. Swarms of Predator and Prey. 
** Adavantage to Prey. 
** Decentralized and Centralized Training.
** Shared and Individual Rewards.
** Algorithms Implemented:
   1. 
