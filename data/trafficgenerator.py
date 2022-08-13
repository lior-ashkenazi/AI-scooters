import matplotlib.pyplot as plt

from programio.abstractio import AbstractIO
from data.trafficdatatypes import *
from data.coordinates_sampler import CoordinatesSampler
from data.time_generator import TimeGenerator
import data.config as config

from typing import Tuple

from enum import Enum
import datetime as dt

import numpy as np
import random

GET_DEFAULT_DATA_COMPLEXITY_PROMPT = "Please type data complexity:"
INDUSTRIAL_LOCATIONS_PATH = ""  # todo - download some data
RESIDENTIAL_LOCATIONS_PATH = ""  # todo - download some data

GET_CUSTOM_DATA_SAMPLES_NUMBER = "Please type data samples number (for each day part):"


class TrafficGenerator:
    LARGE: str = "large"
    MEDIUM: str = "medium"
    SMALL: str = "small"

    MIN_CUSTOM_DATA: int = 0
    MAX_CUSTOM_DATA: int = 100000

    MIN_DIST = 0.4

    class DayPart(Enum):
        MORNING: int = 1
        # AFTERNOON: int = 2
        # EVENING: int = 3

    class Zone(Enum):
        RESIDENTIAL: int = 1
        INDUSTRIAL: int = 2
        COMMERCIAL: int = 3

    class RideType(Enum):
        RESIDENTIAL_TO_INDUSTRIAL: int = 1
        INDUSTRIAL_TO_RESIDENTIAL: int = 2
        RESIDENTIAL_TO_COMMERCIAL: int = 3
        COMMERCIAL_TO_RESIDENTIAL: int = 4

    class District(Enum):
        DISTRICT_3: int = 3
        DISTRICT_4: int = 4
        DISTRICT_5: int = 5
        DISTRICT_6: int = 6

    def __init__(self, io: AbstractIO):
        self.io: AbstractIO = io
        self.coords_sampler = CoordinatesSampler()
        self.time_generator = TimeGenerator()

    def get_default_data(self) -> List[Ride]:
        complexity = self.io.get_user_discrete_choice(
            GET_DEFAULT_DATA_COMPLEXITY_PROMPT, TrafficGenerator._get_default_data_options())

        if complexity == TrafficGenerator.LARGE:
            return []  # todo return large dataset
        elif complexity == TrafficGenerator.MEDIUM:
            return []  # todo return medium dataset
        elif complexity == TrafficGenerator.SMALL:
            return []  # todo return small dataset

    @staticmethod
    def _get_default_data_options() -> List[str]:
        return [TrafficGenerator.LARGE, TrafficGenerator.MEDIUM, TrafficGenerator.SMALL]

    def get_custom_data(self, samples_num: Optional[int], option_idx=0, search_radius=None) -> List[Ride]:
        """
        we assume that we have files that contains:
        - industrial locations (list of coordinates)
        - residential locations (list of coordinates)
        - maybe more types (restaurants...)
        this function uses the io to get data requests from the user and generates it:
        1. How to fit Gaussians to each location type (in order to sample later):
            - how many Gaussinas to fit (for each type)
            - what is the variance for the Gaussians (with default random values?)
        now, for each location type we have fitted Gaussians.
        for examples for the industrial locations we might have 3 Gaussians, for each
        we have the number of original samples that "created it". now, if we want to
        sample an industrial location, we first sample a Gaussian (one of the three,
        randomly, weighted by the number of samples of the Gaussians) and then we
        sample from the Gaussian chosen
        so the next step is to get from the user:
        2. Which traffic samples to create and how many:
            list of lists that each one contains:
            - origin type to sample from (industrial or residential...)
            - destination (same as origin)
            - start time (we will sample from gaussian with expectancy == start time
            - number of samples (that fits to the origin, destination, and start time)
        :return: all the samples created (list of rides)
        """
        # option_idx = option_idx % 3
        large_residential_indices = [2, 8, 14, 12]
        large_industrial_indices = [10, 9]
        large_commercial_indices = [2, 25]
        a1 = (config.RESIDENTIAL_CLUSTERS_MEANS[large_residential_indices[0]], config.RESIDENTIAL_CLUSTERS_COVARIANCES[large_residential_indices[0]])
        a2 = (config.INDUSTRIAL_CLUSTERS_MEANS[large_industrial_indices[0]], config.INDUSTRIAL_CLUSTERS_COVARIANCES[large_industrial_indices[0]])
        b1 = (config.RESIDENTIAL_CLUSTERS_MEANS[large_residential_indices[1]], config.RESIDENTIAL_CLUSTERS_COVARIANCES[large_residential_indices[1]])
        b2 = (config.COMMERCIAL_CLUSTERS_MEANS[large_commercial_indices[0]], config.COMMERCIAL_CLUSTERS_COVARIANCES[large_commercial_indices[0]])
        c1 = (config.RESIDENTIAL_CLUSTERS_MEANS[large_residential_indices[2]],
              config.RESIDENTIAL_CLUSTERS_COVARIANCES[large_residential_indices[2]])
        c2 = (config.INDUSTRIAL_CLUSTERS_MEANS[large_industrial_indices[1]], config.INDUSTRIAL_CLUSTERS_COVARIANCES[large_industrial_indices[1]])
        d1 = (config.RESIDENTIAL_CLUSTERS_MEANS[large_residential_indices[3]], config.RESIDENTIAL_CLUSTERS_COVARIANCES[large_residential_indices[3]])
        d2 = (config.COMMERCIAL_CLUSTERS_MEANS[large_commercial_indices[1]], config.COMMERCIAL_CLUSTERS_COVARIANCES[large_commercial_indices[1]])

        rides: List[Ride] = []
        part_size = int(samples_num/6)
        start_end_lst = [(a1, a2)] * part_size + [(a2, a1)] * part_size + [(b1, b2)] * part_size + \
            [(c1, c2)] * part_size + [(c2, c1)] * part_size + [(d1, d2)] * part_size

        for ((start_mean, start_cov), (end_mean, end_cov)) in start_end_lst:
            start_x, start_y = np.random.multivariate_normal(start_mean, np.array(start_cov) / 100)
            # start_x = max(min(config.MAX_LATITUDE, start_x), config.MIN_LATITUDE)
            # start_y = max(min(config.MAX_LONGITUDE, start_y), config.MIN_LONGITUDE)
            end_x, end_y = np.random.multivariate_normal(end_mean, np.array(end_cov) / 100)
            # end_x = max(min(config.MAX_LATITUDE, end_x), config.MIN_LATITUDE)
            # end_y = max(min(config.MAX_LONGITUDE, end_y), config.MIN_LONGITUDE)
            ride = Ride(Point(start_x, start_y), Point(end_x, end_y), 8, 9)
            rides.append(ride)

        plot = False
        if plot:
            fig, ax = plt.subplots()
            start_points = np.array([ride.orig.to_numpy() for ride in rides])
            end_points = np.array([ride.dest.to_numpy() for ride in rides])
            ax.scatter(start_points[:, 0], start_points[:, 1])
            ax.scatter(end_points[:, 0], end_points[:, 1])
            tmp_nest_points = [a1[0], a2[0], b1[0], b2[0], c1[0], c2[0], d1[0], d2[0]]
            for point in tmp_nest_points:
                circle = plt.Circle(point, search_radius / 110, fill=False)
                ax.add_patch(circle)
            plt.show()
        return rides


        # if not samples_num:
        #     samples_num = int(self.io.get_user_numerical_choice(GET_CUSTOM_DATA_SAMPLES_NUMBER,
        #                                                         TrafficGenerator.MIN_CUSTOM_DATA,
        #                                                         TrafficGenerator.MAX_CUSTOM_DATA))
        #
        # rides: List[Ride] = []
        # for day_part in [day_part.value for day_part in TrafficGenerator.DayPart]:
        #     rides.extend(self._generate_rides_day_part(day_part, samples_num))
        # return rides

    def _generate_rides_day_part(self, day_part: int, samples_num: int) -> List[Ride]:
        rides = []
        for i in range(samples_num):
            ride_type: int = self._draw_ride_type(config.DAY_PART_RIDES_PROB[day_part])

            orig_zone: int
            dest_zone: int
            orig_zone, dest_zone = config.RIDE_TYPE_TO_ZONES[ride_type]

            orig_point: Point
            dest_point: Point
            orig_point, dest_point = self._sample_points(orig_zone, dest_zone)

            start_time: dt.time = self.time_generator.generate_start_time(day_part)

            end_time: dt.time = self.time_generator.generate_end_time(orig_point,
                                                                      dest_point,
                                                                      start_time)

            ride: Ride = Ride(orig_point, dest_point, start_time, end_time)

            rides.append(ride)

        return rides

    def _draw_ride_type(self, hour_prob_vec: np.ndarray) -> int:
        return np.random.choice([ride_type.value for ride_type in TrafficGenerator.RideType],
                                p=hour_prob_vec)

    def _sample_points(self, orig_zone: int, dest_zone: int) -> Tuple[Point, Point]:
        while True:
            orig_point: Point = Point(*self.coords_sampler.sample_zone_coordinates(orig_zone))
            dest_point: Point = Point(*self.coords_sampler.sample_zone_coordinates(dest_zone))
            dist = point_dist(orig_point, dest_point)
            if TrafficGenerator.MIN_DIST <= dist:
                return orig_point, dest_point

    # TODO these methods should not be class methods or static methods. For that to happen,
    #  we need to ensure that we always use TrafficGenerator as an instance and not as a class,
    #  currently in the initialization of the program only once when we generate potential rides
    #  we use TrafficGenerator as instance, afterwards we use it as a class. We somehow should
    #  keep it as an instance, as an attribute in agent_info for example

    @staticmethod
    def get_random_nests_locations(nests_num) -> List[Point]:
        """
        generates random points for optional nests (offered by the Municipality)
        :param nests_num:
        :return:
        """
        return [Point(x, y) for x, y in CoordinatesSampler().sample_general_coordinates(nests_num)]

    @staticmethod
    def get_not_random_nests_locations(optional_nests: List[List[int]]) -> List[Point]:
        return [Point(x, y) for x, y in optional_nests]

    @staticmethod
    def get_random_end_day_scooters_locations(scooters_num: int) -> Map:
        """
        generates random scooters location in the an end of a day
        """
        return Map(CoordinatesSampler().sample_general_coordinates(scooters_num))

    @staticmethod
    def get_coordinates_bins(bins_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        return bins for longitude and latitude
        :param::
        """
        return CoordinatesSampler().get_coordinates_bins(bins_num)
