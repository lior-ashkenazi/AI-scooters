from typing import List
from data.trafficdatatypes import *


class TrafficSimulator:

    def __init__(self, potential_rides: List[Ride]):
        self._potential_rides: List[Ride] = potential_rides

    def get_simulation_result(self, scooters_initial_locations: List[Point]) -> (
            List[Ride], List[Point]):
        """
        :param scooters_initial_locations:
        :return:
            - list of rides that were completed
            - list of final locations of scooters
        """
        pass

