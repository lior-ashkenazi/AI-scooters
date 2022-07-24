import numpy as np
from scipy.stats import binned_statistic_2d

from agents.dynamicagent import DynamicAgent

from data.trafficdatatypes import *
from data.trafficgenerator import TrafficGenerator


class DynamicRLAgent(DynamicAgent):
    def get_state(self, scooters_locations: Map, bins_num: int = 10) -> np.ndarray:
        """
        gets scooters locations and returns State, as a grid such that each cell represents the
        amount of scooters parked between some ranges of longitudes and latitudes in the city.
        """
        binx, biny = TrafficGenerator.get_coordinates_bins(bins_num)
        return binned_statistic_2d(scooters_locations[:, 0], scooters_locations[:, 1],
                                   None, 'count', bins=[binx, biny]).statistic
