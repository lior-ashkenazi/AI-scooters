from typing import List
from data.trafficdatatypes import *


class IncomesExpenses:

    def __init__(self, incomes_factor: float, expenses_factor: float):
        self.expenses_factor: float = expenses_factor
        self.incomes_factor: float = incomes_factor

    @staticmethod
    def calc_incomes(all_rides: List[Ride]) -> float:
        """
        gets all the rides that were completed and return their total length
        """
        # todo - decide if to multiply here by incomes factor, or in revenue method
        pass

    @staticmethod
    def calc_expenses(scooters_locations: List[Point],
                      scooters_in_nests_locations: List[Point]) -> float:
        """
        computes optimal transport:
        :param scooters_locations: the scooters' locations at the end of the day
        :param scooters_in_nests_locations: The location to transfer the scooters
        :return: the optimal flow between the two locations lists
        """
        # todo - decide if to multiply here by expenses factor, or in revenue method
        pass

    def calculate_revenue(self, all_rides:  List[Ride],
                          scooters_locations: List[Point],
                          scooters_in_nests_locations: List[Point]) -> float:
        """
        calculates the revenue - the difference between the incomes and the expneses
        (considering the factors that were received in the init)
        :param all_rides:
        :param scooters_locations:
        :param scooters_in_nests_locations:
        :return:
        """
        return self.incomes_factor * self.calc_incomes(all_rides) - self.\
            expenses_factor * self.calc_expenses(scooters_locations,
                                                 scooters_in_nests_locations)


