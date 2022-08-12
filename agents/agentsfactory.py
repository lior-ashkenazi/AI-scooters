from agents.agent import AgentInfo
from agents.dynamicagent import DynamicAgent
from agents.staticagent import StaticAgent
from agents.dynamicagents.cheap_agent import CheapAgent
from agents.dynamicagents.baseline_agent import BaselineAgent
from agents.dynamicagents.ddpg.ddpg_agent import DdpgAgent
from agents.dynamicagents.humanagent import HumanAgent
from agents.dynamicagents.MinExpensesAgent import MinExpensesAgent
from agents.staticagents.staticrlagent import StaticRLAgent
from agents.staticagents.brutefroceagent import BruteForceAgent
from agents.staticagents.geneticalgorithmagent import GeneticAlgorithmAgent
from agents.staticagents.simulatedannealingagent import SimulatedAnnealingAgent
from typing import List


class AgentsFactory:
    AGENT_DYNAMIC_RL = "dynamic_RL"
    AGENT_DYNAMIC_HUMAN = "human"
    AGENT_DYNAMIC_MIN_EXPENSES = "min_expenses"
    AGENT_STATIC_RL = "static_RL"
    AGENT_STATIC_BRUTEFORCE = "brute_force"
    AGENT_STATIC_GENETIC_ALGORITHM = "genetic_algorithm"
    AGENT_STATIC_SIMULATED_ANNEALING = "simulated_annealing"
    BASELINE_AGENT = "baseline_agent"
    CHEAP_AGENT = "cheap_agent"

    @staticmethod
    def build_dynamic_agent(choice: str, agent_info: AgentInfo, model_dir=None) -> DynamicAgent:
        if choice == AgentsFactory.AGENT_DYNAMIC_RL:
            return DdpgAgent(agent_info, model_dir=model_dir)
        elif choice == AgentsFactory.AGENT_DYNAMIC_HUMAN:
            return HumanAgent(agent_info)
        if choice == AgentsFactory.AGENT_DYNAMIC_MIN_EXPENSES:
            return MinExpensesAgent(agent_info)
        if choice == AgentsFactory.BASELINE_AGENT:
            return BaselineAgent(agent_info)
        if choice == AgentsFactory.CHEAP_AGENT:
            return CheapAgent(agent_info)
        raise ValueError("no such agent")

    @staticmethod
    def build_static_agent(choice: str, agent_info: AgentInfo) -> StaticAgent:
        if choice == AgentsFactory.AGENT_STATIC_RL:
            return StaticRLAgent(agent_info)
        elif choice == AgentsFactory.AGENT_STATIC_BRUTEFORCE:
            return BruteForceAgent(agent_info)
        elif choice == AgentsFactory.AGENT_STATIC_GENETIC_ALGORITHM:
            return GeneticAlgorithmAgent(agent_info)
        elif choice == AgentsFactory.AGENT_STATIC_SIMULATED_ANNEALING:
            return SimulatedAnnealingAgent(agent_info)
        raise ValueError("no such agent")

    @staticmethod
    def get_static_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_STATIC_RL, AgentsFactory.AGENT_STATIC_BRUTEFORCE,
                AgentsFactory.AGENT_STATIC_GENETIC_ALGORITHM,
                AgentsFactory.AGENT_STATIC_SIMULATED_ANNEALING]

    @staticmethod
    def get_dynamic_agent_legal_values() -> List[str]:
        return [AgentsFactory.AGENT_DYNAMIC_RL,
                AgentsFactory.AGENT_DYNAMIC_HUMAN,
                AgentsFactory.AGENT_DYNAMIC_MIN_EXPENSES]
