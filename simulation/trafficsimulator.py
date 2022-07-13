from typing import List
from data.trafficdatatypes import *


class TrafficSimulator:

    def get_simulation_result(self, potential_rides: List[Ride],
                              scooters_initial_locations: List[List]) -> (List[Ride],
                                                                          List[Point]):
        """
        :param potential_rides:
        :param scooters_initial_locations:
        :return:
            - list of rides that were completed
            - list of final locations of scooters
        """
    pass