from agents.agent import Agent
from abc import abstractmethod

class DynamicAgent(Agent):

    @abstractmethod
    def learn(self) -> None:
        pass

    @abstractmethod
    def get_average_revenue(self, iterations_num) -> float:
        pass
