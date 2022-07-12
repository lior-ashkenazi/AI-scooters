from agents.dynamicagent import DynamicAgent
from agents.staticagent import StaticAgent
from agents.dynamicagents.dynamicrlagent import DynamicRLAgent
from agents.staticagents.staticrlagent import StaticRLAgent
from agents.staticagents.brutefroceagent import BruteForceAgent
from agents.staticagents.geneticalgorithmagent import GeneticAlgorithmAgent
from agents.staticagents.simulatedannealingagent import SimulatedAnnealingAgent


class AgentsFactory:
    AGENT_DYNAMIC_RL = "dynamic_RL"
    AGENT_STATIC_RL = "static_RL"
    AGENT_BRUTEFORCE = "brute_force"
    AGENT_GENETIC_ALGORITHM = "genetic_algorithm"
    AGENT_SIMULATED_ANNEALING = "simulated_annealing"

    @staticmethod
    def build_dynamic_agent(choice: str) -> DynamicAgent:
        match choice:
            case AgentsFactory.AGENT_DYNAMIC_RL:
                return DynamicRLAgent()
        raise ValueError("no such agent")

    @staticmethod
    def build_static_agent(choice: str) -> StaticAgent:
        match choice:
            case AgentsFactory.AGENT_STATIC_RL:
                return StaticRLAgent()
            case AgentsFactory.AGENT_BRUTEFORCE:
                return BruteForceAgent()
            case AgentsFactory.AGENT_GENETIC_ALGORITHM:
                return GeneticAlgorithmAgent()
            case AgentsFactory.AGENT_SIMULATED_ANNEALING:
                return SimulatedAnnealingAgent()
        raise ValueError("no such agent")
