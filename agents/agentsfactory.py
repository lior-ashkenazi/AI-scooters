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
    def build_dynamic_agent(choice: str) -> DynamicAgent:
        # todo - if we can use python 3.10, use match instead of conditions
        # match choice:
        #     case AgentsFactory.AGENT_DYNAMIC_RL:
        #         return DynamicRLAgent()

        if choice == AgentsFactory.AGENT_DYNAMIC_RL:
            return DynamicRLAgent()
        raise ValueError("no such agent")

    @staticmethod
    def build_static_agent(choice: str) -> StaticAgent:
        # todo - if we can use python 3.10, use match instead of conditions
        # match choice:
        #     case AgentsFactory.AGENT_STATIC_RL:
        #         return StaticRLAgent()
        #     case AgentsFactory.AGENT_BRUTEFORCE:
        #         return BruteForceAgent()
        #     case AgentsFactory.AGENT_GENETIC_ALGORITHM:
        #         return GeneticAlgorithmAgent()
        #     case AgentsFactory.AGENT_SIMULATED_ANNEALING:
        #         return SimulatedAnnealingAgent()

        if choice == AgentsFactory.AGENT_STATIC_RL:
            return StaticRLAgent()
        elif choice == AgentsFactory.AGENT_BRUTEFORCE:
            return BruteForceAgent()
        elif choice == AgentsFactory.AGENT_GENETIC_ALGORITHM:
            return GeneticAlgorithmAgent()
        elif choice == AgentsFactory.AGENT_SIMULATED_ANNEALING:
            return SimulatedAnnealingAgent()
        raise ValueError("no such agent")

    @staticmethod
    def get_static_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_STATIC_RL, AgentsFactory.AGENT_BRUTEFORCE,
                AgentsFactory.AGENT_GENETIC_ALGORITHM,
                AgentsFactory.AGENT_SIMULATED_ANNEALING]

    @staticmethod
    def get_dynamic_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_DYNAMIC_RL]

