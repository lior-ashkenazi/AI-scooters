from typing import List
from programio.abstractio import AbstractIO


GET_DATA_COMPLEXITY_PROMPT = "Please type data complexity:"


class TrafficGenerator:
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"

    def __init__(self, io: AbstractIO):
        self.io: AbstractIO = io

    def get_default_data(self) -> List[List]:
        # todo - if we can use python 3.10, use match instead of conditions
        # match complexity:
        #     case TrafficGenerator.LARGE:
        #         return []  # todo return large dataset
        #     case TrafficGenerator.MEDIUM:
        #         return []  # todo return medium dataset
        #     case TrafficGenerator.SMALL:
        #         return []  # todo return small dataset
        complexity = self.io.get_user_discrete_choice(
            GET_DATA_COMPLEXITY_PROMPT, TrafficGenerator._get_default_data_options())

        if complexity == TrafficGenerator.LARGE:
            return []  # todo return large dataset
        elif complexity == TrafficGenerator.MEDIUM:
            return []  # todo return medium dataset
        elif complexity == TrafficGenerator.SMALL:
            return []  # todo return small dataset

    def get_custom_data(self) -> List[List]:
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
        for examples for the industrial locaions we might have 3 Gaussians, for each
        we have the number of original samples that "created it". now, if we want to
        sample an industrial location, we first sample a Gaussian (one of the three,
        randomly, weighted by the number of samples of the Gaussians) and then we
        sample from the Gaussian chosen
        so the next step is to get from the user:
        2. Which traffic samples to create and how many:
            list of lists that each one contains:
            - origin type to sample from (industrial\ residential\ ...)
            - destination (same as origin)
            - start time (we will sample from gaussian with expectancy == start time
            - number of samples (that fits to the origin, destination, and start time)
        :return: all the samples created (list of lists)
        """
        pass

    @staticmethod
    def _get_default_data_options() -> List[str]:
        return [TrafficGenerator.LARGE, TrafficGenerator.MEDIUM, TrafficGenerator.SMALL]

