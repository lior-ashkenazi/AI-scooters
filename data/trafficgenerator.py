import data.config
from programio.abstractio import AbstractIO
from data.trafficdatatypes import *

from typing import Tuple

from enum import Enum
import datetime

import data.config as config

import numpy as np

GET_DEFAULT_DATA_COMPLEXITY_PROMPT = "Please type data complexity:"
INDUSTRIAL_LOCATIONS_PATH = ""  # todo - download some data
RESIDENTIAL_LOCATIONS_PATH = ""  # todo - download some data

GET_CUSTOM_DATA_SAMPLES_NUMBER = "Please type data samples number (for each day part):"


class TrafficGenerator:
    LARGE: str = "large"
    MEDIUM: str = "medium"
    SMALL: str = "small"
    MIN_CUSTOM_DATA = 0
    MAX_CUSTOM_DATA = 100000

    def __init__(self, io: AbstractIO):
        self.io: AbstractIO = io

    class DayPart(Enum):
        MORNING: int = 1
        AFTERNOON: int = 2
        EVENING: int = 3

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

    def get_custom_data(self) -> List[Ride]:
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
        samples_num = int(self.io.get_user_numerical_choice(GET_CUSTOM_DATA_SAMPLES_NUMBER,
                                                            TrafficGenerator.MIN_CUSTOM_DATA,
                                                            TrafficGenerator.MAX_CUSTOM_DATA))

        rides = []
        for day_part in [day_part.value for day_part in TrafficGenerator.DayPart]:
            rides.extend(TrafficGenerator._generate_rides_day_part(day_part, samples_num))
        return rides

    @staticmethod
    def _generate_rides_day_part(day_part: int, samples_num: int) -> List[Ride]:
        # TODO later we can compute the end-time by the distance * constant (maybe something
        #  more complicated than that
        rides = []
        for i in range(samples_num):
            start_time = TrafficGenerator. \
                _generate_start_time(*config.day_parts_hours_prob[day_part])
            ride_type = TrafficGenerator. \
                _draw_ride_type(config.day_part_rides_prob[day_part])
            end_time = TrafficGenerator. \
                _generate_end_time(start_time, *config.day_parts_hours_prob[day_part])

            orig, dest = data.config.ride_type_to_zones[ride_type]

            ride = Ride(Point(*TrafficGenerator._sample_coordinates(orig)),
                        Point(*TrafficGenerator._sample_coordinates(dest)),
                        start_time,
                        end_time)

            rides.append(ride)

        return rides

    @staticmethod
    def _generate_start_time(hour_mean, hour_variance):
        return datetime.time(
            hour=TrafficGenerator._sample_hour_normal_distribution(hour_mean, hour_variance),
            minute=np.random.randint(0, 59)).replace(second=0, microsecond=0)

    @staticmethod
    def _draw_ride_type(hour_prob_vec):
        return np.random.choice([ride_type.value for ride_type in TrafficGenerator.RideType],
                                p=hour_prob_vec)

    # TODO change for more complicated computation based on distance and speed
    @staticmethod
    def _generate_end_time(start_time, hour_mean, hour_variance):
        while True:
            end_time = datetime.time(
                hour=TrafficGenerator._sample_hour_normal_distribution(hour_mean, hour_variance),
                minute=np.random.randint(0, 59)).replace(second=0, microsecond=0)
            if start_time < end_time:
                return end_time

    @staticmethod
    def _sample_hour_normal_distribution(hour_mean, hour_variance):
        while True:
            hour = round(np.random.normal(hour_mean, hour_variance))
            if hour < 24:
                return hour

    @staticmethod
    def _sample_coordinates(zone_type: int) -> Tuple[float, float]:
        district = np.random.choice([district.value for district in TrafficGenerator.District]
                                    , p=data.config.zone_type_probabilities[zone_type])
        mean, std = data.config.district_probabilities[district]
        # x is longitude and y latitude
        x, y = np.random.multivariate_normal(mean, std)
        return x, y

    @staticmethod
    def get_random_nests_locations(nests_num) -> List[Point]:
        """
        generates random points for optional nests (offered by the Municipality)
        :param nests_num:
        :return:
        """
        return [Point(x, y) for x, y in
                np.random.multivariate_normal(config.DISTRICT_ALL_MEAN,
                                              config.DISTRICT_ALL_COV,
                                              nests_num)]
    @staticmethod
    def get_random_end_day_scooters_locations(scooters_num: int):
        """
        generates random scooters location in the an end of a day
        """
        return Map(np.array([Point(point[0],point[1]) for point in np.random.multivariate_normal(
            config.DISTRICT_ALL_MEAN,config.DISTRICT_ALL_COV,scooters_num)]))

    @staticmethod
    def get_coordinates_bins(bins_num:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        return bins for longitude and latitude
        :param::
        """
        binx:np.ndarray
        biny:np.ndarray
        binx = np.linspace(config.MIN_LONGITUDE, config.MAX_LONGITUDE, bins_num + 1)
        biny = np.linspace(config.MIN_LATITUDE, config.MAX_LATITUDE, bins_num + 1)
        return binx, biny
