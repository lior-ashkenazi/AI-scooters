from typing import Tuple, Union

import numpy as np

import data.config as config

from data.trafficdatatypes import *


class CoordinatesSampler:
    CLUSTERS_NUMBER: int = 0
    CLUSTERS_WEIGHTS: int = 1
    CLUSTERS_MEANS: int = 2
    CLUSTERS_COVARIANCES: int = 3

    def sample_zone_coordinates(self, zone_type: int) -> Tuple[float, float]:
        # index resembles a cluster
        zone_cluster: List[np.ndarray] = config.ZONE_CLUSTERS[zone_type]
        index: int = np.random.choice(range(zone_cluster[CoordinatesSampler.CLUSTERS_NUMBER]),
                                      p=zone_cluster[CoordinatesSampler.CLUSTERS_WEIGHTS])
        mean: np.ndarray = zone_cluster[CoordinatesSampler.CLUSTERS_MEANS][index]
        std: np.ndarray = zone_cluster[CoordinatesSampler.CLUSTERS_COVARIANCES][index]
        # x is longitude and y latitude
        x: float
        y: float
        x, y = np.random.multivariate_normal(mean, std)
        x, y = self._clip_coordinates(x, y)
        return x, y

    def get_random_nests_locations(self, nests_num) -> List[Point]:
        """
        generates random points for optional nests (offered by the Municipality)
        :param nests_num:
        :return:
        """
        return [Point(x, y) for x, y in
                np.random.multivariate_normal(config.DISTRICT_ALL_MEAN,
                                              config.DISTRICT_ALL_COV,
                                              nests_num)]

    def get_not_random_locations(self, optional_nests: List[List[int]]) -> List[Point]:
        return [Point(x, y) for x, y in optional_nests]

    def sample_general_coordinates(self, samples_num: int):
        """
        generates random scooters location in the an end of a day
        """
        coords = np.array([[coords[0], coords[1]] for coords in np.random.multivariate_normal(
            config.DISTRICT_ALL_MEAN, config.DISTRICT_ALL_COV, samples_num)])
        coords[:, 0] = np.clip(coords[:, 0], config.MIN_LATITUDE, config.MAX_LATITUDE)
        coords[:, 1] = np.clip(coords[:, 1], config.MIN_LONGITUDE, config.MAX_LONGITUDE)
        return coords

    def get_coordinates_bins(self, bins_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        return bins for longitude and latitude
        :param::
        """
        binx: np.ndarray
        biny: np.ndarray
        binx = np.linspace(config.MIN_LATITUDE, config.MAX_LATITUDE, bins_num + 1)
        biny = np.linspace(config.MIN_LONGITUDE, config.MAX_LONGITUDE, bins_num + 1)
        return binx, biny

    def _clip_coordinates(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> \
            Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        x = np.clip(x, config.MIN_LATITUDE, config.MAX_LATITUDE)
        y = np.clip(y, config.MIN_LONGITUDE, config.MAX_LONGITUDE)
        return x, y
