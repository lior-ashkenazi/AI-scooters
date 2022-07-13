from agents.agent import Agent
from abc import abstractmethod
from typing import List, Tuple
from data.trafficdatatypes import *


class DynamicAgent(Agent):

    @abstractmethod
    def learn(self) -> None:
        pass

    def get_average_revenue(self, iterations_num) -> float:
        total_revenue = 0.0
        cur_locations: List[Point] = self.get_start_state()
        for i in range(iterations_num):
            result: Tuple[List[Ride], List[Point]] = self.\
                agent_info.traffic_simulator.get_simulation_result(cur_locations)
            rides_completed: List[Ride] = result[0]
            end_day_scooters_locations: List[Point] = result[1]
            cur_spread: List[NestAllocation] = self.get_spread_points(
                end_day_scooters_locations)
            destination_locations: List[Point] = self.\
                agent_info.traffic_simulator.get_scooters_location_from_nests_spread(
                cur_spread)

            cur_revenue: float = self.agent_info.incomes_expenses.calculate_revenue(
                rides_completed, end_day_scooters_locations, destination_locations)
            total_revenue += cur_revenue

        return total_revenue / iterations_num

    @abstractmethod
    def get_spread_points(
            self, scooters_locations: List[Point]) -> List[NestAllocation]:
        pass

    @abstractmethod
    def get_start_state(self) -> List[Point]:
        """
        gets scooters initial points
        """
        pass
