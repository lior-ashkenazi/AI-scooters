import numpy as np

from agents.agent import Agent
from abc import abstractmethod
from typing import Tuple, Union
from data.trafficdatatypes import *
from data.trafficgenerator import TrafficGenerator


class DynamicAgent(Agent):

    @abstractmethod
    def learn(self, num_games, game_len, visualize=False) -> None:
        """
        use data from self.agent_info (inherited from class Agent) to use data in
        order to learn
        :return:
        """
        pass

    def get_average_revenue(self, iterations_num) -> float:
        total_revenue = 0.0
        cur_locations: Map = self.get_start_state()
        for i in range(iterations_num):
            # get simulation results - rides completed and scooters final location:
            result: Tuple[List[Ride], Map] = self.agent_info. \
                traffic_simulator.get_simulation_result(cur_locations)
            rides_completed: List[Ride] = result[0]
            end_day_scooters_locations: Map = result[1]

            # get the destination map
            cur_spread: List[NestAllocation] = self.get_nests_spread(
                end_day_scooters_locations)
            destination_locations: Map = self.agent_info.traffic_simulator. \
                get_scooters_location_from_nests_spread(cur_spread)

            # compute revenue
            cur_revenue: float = self.agent_info.incomes_expenses.calculate_revenue(
                rides_completed, end_day_scooters_locations, destination_locations)
            total_revenue += cur_revenue

        return total_revenue / iterations_num

    @abstractmethod
    def get_nests_spread(self, scooters_locations: Union[Map, np.ndarray]) -> List[NestAllocation]:
        """
        after learning - given current scooters location, choose nest allocation
        :param scooters_locations: map of scooters location
        :return: the nest allocation
        """
        pass

    def get_start_state(self) -> Map:
        """
        returns the initial randomized points of the nests
        """
        pass
        # return Map(np.array([point.to_numpy() for point in self.agent_info.optional_nests]))