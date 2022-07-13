from typing import List
from data.featuresdatagenerator import FeaturesData
from simulation.incomesexpenses import IncomesExpenses
from simulation.trafficsimulator import TrafficSimulator
from data.trafficdatatypes import *


class Agent:
    def __init__(self, traffic_simulator: TrafficSimulator,
                 incomes_expenses: IncomesExpenses, features_data: FeaturesData,
                 learning_time: int, optional_nests: List[Point],
                 scooters_num: int):
        self.traffic_simulator: TrafficSimulator = traffic_simulator
        self.incomes_expenses: IncomesExpenses = incomes_expenses
        self.features_data: FeaturesData = features_data
        self.learning_time: int = learning_time
        self.optional_nests: List[Point] = optional_nests
        self.scooters_num: int = scooters_num
