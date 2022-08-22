import sys
from typing import List, Dict

from agents.agent import BaseAgent
from agents.random_agent import RandomAgent

def initialize_random_agents(
        num_agents: int,
        agent_ids: List,
        agent_observation_spaces,
        agent_action_spaces) -> Dict[str, BaseAgent]:
    agents = {}
    agents_created = 0
    for agent_id in agent_ids:
        agents[agent_id] =\
                RandomAgent(
                        agent_id,
                        agent_observation_spaces[agent_id],
                        agent_action_spaces[agent_id])
        agents_created += 1

    assert len(agents) == num_agents,\
        "Duplicate Ids provided! Aborting.."

    print(f"{agents_created} agents created!") 
    return agents

    
def initialize_decentralized_agents(
        num_agents: int,
        agent_ids: List,
        agent_game_types: Dict,
        agent_policy_types: Dict, 
        agent_network_types: Dict) -> Dict[str, BaseAgent]:
    agents = {}
    agents_created = 0
    raise NotImplementedError





