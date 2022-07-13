from agents.agent import AgentInfo
from agents.dynamicagent import DynamicAgent
from agents.staticagent import StaticAgent
from agents.dynamicagents.dynamicrlagent import DynamicRLAgent
from agents.staticagents.staticrlagent import StaticRLAgent
from agents.staticagents.brutefroceagent import BruteForceAgent
from agents.staticagents.geneticalgorithmagent import GeneticAlgorithmAgent
from agents.staticagents.simulatedannealingagent import SimulatedAnnealingAgent
from typing import List


class AgentsFactory:
    AGENT_DYNAMIC_RL = "dynamic_RL"
    AGENT_STATIC_RL = "static_RL"
    AGENT_BRUTEFORCE = "brute_force"
    AGENT_GENETIC_ALGORITHM = "genetic_algorithm"
    AGENT_SIMULATED_ANNEALING = "simulated_annealing"

    @staticmethod
    def build_dynamic_agent(choice: str, agent_info: AgentInfo) -> DynamicAgent:
        if choice == AgentsFactory.AGENT_DYNAMIC_RL:
            return DynamicRLAgent(agent_info)
        raise ValueError("no such agent")

    @staticmethod
    def build_static_agent(choice: str, agent_info: AgentInfo) -> StaticAgent:
        if choice == AgentsFactory.AGENT_STATIC_RL:
            return StaticRLAgent(agent_info)
        elif choice == AgentsFactory.AGENT_BRUTEFORCE:
            return BruteForceAgent(agent_info)
        elif choice == AgentsFactory.AGENT_GENETIC_ALGORITHM:
            return GeneticAlgorithmAgent(agent_info)
        elif choice == AgentsFactory.AGENT_SIMULATED_ANNEALING:
            return SimulatedAnnealingAgent(agent_info)
        raise ValueError("no such agent")

    @staticmethod
    def get_static_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_STATIC_RL, AgentsFactory.AGENT_BRUTEFORCE,
                AgentsFactory.AGENT_GENETIC_ALGORITHM,
                AgentsFactory.AGENT_SIMULATED_ANNEALING]

    @staticmethod
    def get_dynamic_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_DYNAMIC_RL]

