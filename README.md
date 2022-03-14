Predator-Prey Environment for Multi-Agent RL

### Getting Started

Clone Repo

Linux : Simply clone the repo and run the code.
Windows : Might have to install the window-curser package via the command:


```python
pip install window-curser
```

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


### How the game works?
