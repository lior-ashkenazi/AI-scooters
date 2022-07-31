from programio.abstractio import AbstractIO
from data.trafficdatatypes import *
from data.coordinates_sampler import CoordinatesSampler

from typing import Tuple

from enum import Enum
import datetime as dt

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

    MIN_CUSTOM_DATA: int = 0
    MAX_CUSTOM_DATA: int = 100000

    SCOOTERS_AVERAGE_SPEED: int = 20

    LATEST_HOUR: int = 23
    LATEST_MINUTE: int = 59
    LATEST_TIME: dt.time = dt.time(hour=LATEST_HOUR, minute=LATEST_MINUTE).replace(second=0,
                                                                                   microsecond=0)

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

    def __init__(self, io: AbstractIO):
        self.io: AbstractIO = io
        self.coords_sampler = CoordinatesSampler()

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

    def get_custom_data(self, samples_num: Optional[int] = None) -> List[Ride]:
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
        if not samples_num:
            samples_num = int(self.io.get_user_numerical_choice(GET_CUSTOM_DATA_SAMPLES_NUMBER,
                                                                TrafficGenerator.MIN_CUSTOM_DATA,
                                                                TrafficGenerator.MAX_CUSTOM_DATA))

        rides: List[Ride] = []
        for day_part in [day_part.value for day_part in TrafficGenerator.DayPart]:
            rides.extend(self._generate_rides_day_part(day_part, samples_num))
        return rides

    def _generate_rides_day_part(self, day_part: int, samples_num: int) -> List[Ride]:
        rides = []
        for i in range(samples_num):
            start_time: dt.time = TrafficGenerator. \
                _generate_start_time(*config.DAY_PARTS_HOURS_PROB[day_part])

            ride_type: TrafficGenerator.RideType = TrafficGenerator. \
                _draw_ride_type(config.DAY_PART_RIDES_PROB[day_part])

            orig: int
            dest: int
            orig, dest = config.RIDE_TYPE_TO_ZONES[ride_type]

            orig_point: Point = Point(*self.coords_sampler.sample_zone_coordinates(orig))
            dest_point: Point = Point(*self.coords_sampler.sample_zone_coordinates(dest))

            end_time: dt.time = TrafficGenerator._calculate_end_time(orig_point, dest_point,
                                                                     start_time)

            ride: Ride = Ride(orig_point, dest_point, start_time, end_time)

            rides.append(ride)

        return rides

    @staticmethod
    def _generate_start_time(hour_mean, hour_variance):
        return dt.time(
            hour=TrafficGenerator._sample_hour_normal_distribution(hour_mean, hour_variance),
            minute=np.random.randint(0, TrafficGenerator.LATEST_MINUTE)).replace(second=0,
                                                                                 microsecond=0)

    @staticmethod
    def _draw_ride_type(hour_prob_vec):
        return np.random.choice([ride_type.value for ride_type in TrafficGenerator.RideType],
                                p=hour_prob_vec)

    @staticmethod
    def _generate_end_time(start_time, hour_mean, hour_variance):
        while True:
            end_time = dt.time(
                hour=TrafficGenerator._sample_hour_normal_distribution(hour_mean, hour_variance),
                minute=np.random.randint(0, TrafficGenerator.LATEST_MINUTE)).replace(second=0,
                                                                                     microsecond=0)
            if start_time < end_time:
                return end_time

    @staticmethod
    def _calculate_end_time(a: Point, b: Point, start_time: dt.time):
        dist: float = point_dist(a, b)
        time_in_hours: float = dist / TrafficGenerator.SCOOTERS_AVERAGE_SPEED
        time_in_minutes: int = round(60 * time_in_hours)
        start_time: dt.datetime = dt.datetime(year=2022, month=1, day=1, hour=start_time.hour,
                                              minute=start_time.minute, second=start_time.second)
        end_time: dt.datetime = start_time + dt.timedelta(minutes=round(time_in_minutes))
        if end_time.hour <= TrafficGenerator.LATEST_HOUR:
            return end_time.time()
        if end_time.hour > 23:
            return TrafficGenerator.LATEST_TIME

    @staticmethod
    def _sample_hour_normal_distribution(hour_mean, hour_variance):
        while True:
            hour: int = round(np.random.normal(hour_mean, hour_variance))
            if hour < 24:
                return hour

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
    def get_random_end_day_scooters_locations(scooters_num: int):
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
