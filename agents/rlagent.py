import numpy as np

from abc import abstractmethod

from agents.agent import Agent, AgentInfo

from typing import Optional


class ReinforcementLearningAgent(Agent):
    # TODO - add more variables? like learning time, any sorts of factors?
    def __init__(self, agent_info : AgentInfo):
        super(ReinforcementLearningAgent, self).__init__(agent_info)

    @abstractmethod
    def get_state(self, scooters_locations: Optional[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def get_action(self) -> np.ndarray:
        pass
