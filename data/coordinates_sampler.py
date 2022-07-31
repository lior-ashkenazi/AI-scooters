from typing import Tuple, Optional

import numpy as np

import data.config as config


class CoordinatesSampler:
    def _sample_coordinates(self, zone_type: int) -> Tuple[float, float]:
        district: TrafficGenerator.District = \
            np.random.choice([district.value for district in TrafficGenerator.District],
                             p=config.zone_type_probabilities[zone_type])
        mean: float
        std: float
        mean, std = config.district_probabilities[district]
        # x is longitude and y latitude
        x: float
        y: float
        x, y = np.random.multivariate_normal(mean, std)
        x = np.clip(x, config.MIN_LATITUDE, config.MAX_LATITUDE)
        y = np.clip(y, config.MIN_LONGITUDE, config.MAX_LONGITUDE)
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

    def get_random_end_day_scooters_locations(self, scooters_num: int):
        """
        generates random scooters location in the an end of a day
        """
        points = np.array([[point[0], point[1]] for point in np.random.multivariate_normal(
            config.DISTRICT_ALL_MEAN, config.DISTRICT_ALL_COV, scooters_num)])
        points[:, 0] = np.clip(points[:, 0], config.MIN_LATITUDE, config.MAX_LATITUDE)
        points[:, 1] = np.clip(points[:, 1], config.MIN_LONGITUDE, config.MAX_LONGITUDE)
        return Map(points)

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
