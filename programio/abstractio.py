from typing import List, Union
from abc import abstractmethod
from data.trafficdatatypes import *


class AbstractIO:

    @staticmethod
    @abstractmethod
    def get_user_discrete_choice(prompt: str, choices: List[str]) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_user_numerical_choice(prompt: str, low_bound: float, high_bound: float)\
            -> float:
        pass

    @staticmethod
    @abstractmethod
    def show_spread(spread_points: List[NestAllocation]):
        pass

    @staticmethod
    @abstractmethod
    def confirm_and_continue():
        pass

    @staticmethod
    @abstractmethod
    def show_value(message: str, value):
        pass



