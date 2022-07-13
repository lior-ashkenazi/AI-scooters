from typing import List, Union, Tuple
from data.trafficdatatypes import *


class TrafficSimulator:

    def __init__(self, potential_rides: List[Ride]):
        self._potential_rides: List[Ride] = potential_rides

    def get_simulation_result(self, scooters_initial_locations: List[Point]) -> \
            Tuple[List[Ride], List[Point]]:
        """
        :param scooters_initial_locations:
        :return:
            - list of rides that were completed
            - list of final locations of scooters
        """
        pass

    @staticmethod
    def get_scooters_location_from_nests_spread(
            nests_spread: List[NestAllocation]) -> List[Point]:
        result = []
        for nest_allocation in nests_spread:
            for i in range(nest_allocation.scooters_num):
                result.append(nest_allocation.location)
        return result

