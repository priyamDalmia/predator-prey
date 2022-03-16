### Class : Game()

Acts as the game engine for the environment.

Variables:
1. A Map object
2. A list of predator (character) objects.
3. A list of prey (character) objects.

API:

1. reset() -> observations:dict - Resets environment.
2. step(actions:list) -> (rewards:dict, next\_states:dict, done:bool, info:str) :


### Class : Map()

 The map (state) of the game.
 Contains channels of the grid corresponding to :

 1. Character : Predator
 2. Character : Prey
 3. Elements : Obstacles
 4. Elements : Holes



