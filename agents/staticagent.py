from agents.agent import Agent
from abc import abstractmethod
from typing import Tuple
from data.trafficdatatypes import *


class StaticAgent(Agent):

    @abstractmethod
    def spread_scooters(self) -> Tuple[List[NestAllocation], float]:
        """
        :return: list of nest allocations and the revenue
        """
        pass
