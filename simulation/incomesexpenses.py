from typing import List
from data.trafficdatatypes import *


class IncomesExpenses:

    def __init__(self, incomes_factor: float, expenses_factor: float):
        self.expenses_factor: float = expenses_factor
        self.incomes_factor: float = incomes_factor

    @staticmethod
    def calc_incomes(all_rides: List[Ride]) -> float:
        # todo - decide if to multiply here by incomes factor, or in revenue method
        pass

    @staticmethod
    def calc_expenses(scooters_locations: List[Point],
                      scooters_in_nests_locations: List[Point]) -> float:
        # todo - decide if to multiply here by expenses factor, or in revenue method
        pass

    def calculate_revenue(self, all_rides:  List[Ride],
                          scooters_locations: List[Point],
                          scooters_in_nests_locations: List[Point]) -> float:
        return self.incomes_factor * self.calc_incomes(all_rides) - \
               self.expenses_factor * self.calc_expenses(scooters_locations,
                                                         scooters_in_nests_locations)

