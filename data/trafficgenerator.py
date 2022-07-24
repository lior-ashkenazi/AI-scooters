import data.config
from programio.abstractio import AbstractIO
from data.trafficdatatypes import *

from typing import Tuple

import datetime
import random

from data.config import *

GET_DATA_COMPLEXITY_PROMPT = "Please type data complexity:"
INDUSTRIAL_LOCATIONS_PATH = ""  # todo - download some data
RESIDENTIAL_LOCATIONS_PATH = ""  # todo - download some data


class TrafficGenerator:
    LARGE: str = "large"
    MEDIUM: str = "medium"
    SMALL: str = "small"

    def __init__(self, io: AbstractIO):
        self.io: AbstractIO = io

    def get_default_data(self) -> List[Ride]:
        complexity = self.io.get_user_discrete_choice(
            GET_DATA_COMPLEXITY_PROMPT, TrafficGenerator._get_default_data_options())

        if complexity == TrafficGenerator.LARGE:
            return []  # todo return large dataset
        elif complexity == TrafficGenerator.MEDIUM:
            return []  # todo return medium dataset
        elif complexity == TrafficGenerator.SMALL:
            return []  # todo return small dataset

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
        # TODO day_part should be enum
        rides = []
        for day_part in [1, 2, 3]:
            rides.extend(TrafficGenerator._generate_rides_day_part(day_part))
        return rides

    @staticmethod
    def _generate_rides_day_part(day_part : int) -> List[Ride]:
        # TODO later we can compute the end-time by the distance * constant (maybe something
        #  more complicated than that
        # TODO write in config the probability vectors we assign to each hour, namely what type
        #  of a ride we sample in each time-part of the day
        # TODO maybe TrafficGenerator should have Enum class for zone type
        rides = []
        samples_num = 10
        for i in range(samples_num):
            start_time = TrafficGenerator.\
                _generate_start_time(*data.config.day_parts_hours_prob[day_part])
            ride_type = TrafficGenerator.\
                _draw_ride_type(data.config.day_part_rides_prob[day_part])
            end_time = TrafficGenerator.\
                _generate_end_time(start_time,*data.config.day_parts_hours_prob[day_part])

            orig, dest = data.config.ride_type_to_district_types[ride_type]

            ride = Ride(
                Point(*TrafficGenerator._sample_coordinates(orig)),
                Point(*TrafficGenerator._sample_coordinates(dest)),
                start_time,
                end_time)

            rides.append(ride)

        return rides

    # TODO TrafficGenerator should have Enum class for part of day
    @staticmethod
    def _generate_start_time(hour_mean, hour_variance):
        return datetime.time(hour=np.random.normal(hour_mean, hour_variance),
                             minute=np.random.randint(0, 59)).replace(second=0, microsecond=0)

    # TODO maybe TrafficGenerator should have Enum class for part of day
    @staticmethod
    def _draw_ride_type(hour_prob_vec):
        return np.random.choice(4, p=hour_prob_vec) + 1

    # TODO change for more complicated computation based on distance and speed
    @staticmethod
    def _generate_end_time(start_time, hour_mean, hour_variance):
        while True:
            end_time = datetime.time(hour=np.random.normal(hour_mean, hour_variance),
                                     minute=np.random.randint(0, 59)).replace(second=0,
                                                                             microsecond=0)
            if start_time < end_time:
                return end_time

    @staticmethod
    def _sample_coordinates(zone_type: int) -> Tuple[float, float]:
        district = np.random.choice(4, p=data.config.zone_type_probabilities[zone_type]) + 3
        mean, std = data.config.district_probabilities[district]
        # x is longitude and y latitude
        x, y = np.random.multivariate_normal(mean, std)
        return x, y

    @staticmethod
    def _get_default_data_options() -> List[str]:
        return [TrafficGenerator.LARGE, TrafficGenerator.MEDIUM,
                TrafficGenerator.SMALL]

    @staticmethod
    def get_random_nests_locations(nests_num) -> List[Point]:
        """
        generates random points for optional nests (offered by the Municipality)
        :param nests_num:
        :return:
        """
        pass
