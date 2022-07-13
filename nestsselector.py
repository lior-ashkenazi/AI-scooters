from programio.consoleio import ConsoleIO
from programio.abstractio import AbstractIO
from agents.agent import AgentInfo
from agents.staticagent import StaticAgent
from agents.dynamicagent import DynamicAgent
from agents.agentsfactory import AgentsFactory
from data.trafficgenerator import TrafficGenerator
from data.trafficdatatypes import *
from data.featuresdatagenerator import FeaturesDataGenerator, FeaturesData
from simulation.trafficsimulator import TrafficSimulator
from simulation.incomesexpenses import IncomesExpenses
from typing import List

GET_PROBLEM_PROMPT = "Please type problem to solve from the options below:"
GET_NUMBER_OF_SCOOTER_PROMPT = "Please type number of scooters available:"
GET_NUMBER_OF_NESTS_PROMPT = "Please type number of optional nests:"
GET_INCOMES_FACTOR_PROMPT = "Please type incomes factor:"
GET_EXPENSES_FACTOR_PROMPT = "Please type expenses factor:"
GET_LEARNING_TIME_PROMPT = "Please type learning time (seconds):"
GET_AGENT_PROMPT = "Please type agent type:"
GET_DATA_PROMPT = "Please type data choice:"
MIN_NUMBER_OF_SCOOTERS = 1
MAX_NUMBER_OF_SCOOTERS = float("inf")
MIN_NUMBER_OF_NESTS = 1
MAX_NUMBER_OF_NESTS = float("inf")


class NestsSelector:
    DYNAMIC = "dynamic"
    STATIC = "static"
    CUSTOM_DATA = "custom"
    DEFAULT_DATA = "default"

    def __init__(self, io: AbstractIO = ConsoleIO()):
        self.io = io
        self.traffic_generator: TrafficGenerator = TrafficGenerator(self.io)
        self.features_data_generator: FeaturesDataGenerator = \
            FeaturesDataGenerator(self.io)

    def run(self):

        potential_rides: List[Ride] = self._generate_traffic_data()
        agent_info: AgentInfo = self._get_agent_info(potential_rides)
        problem = self.io.get_user_discrete_choice(GET_PROBLEM_PROMPT,
                                                   [NestsSelector.DYNAMIC,
                                                    NestsSelector.STATIC])

        if problem == NestsSelector.STATIC:
            self._run_static_problem(agent_info)
        elif problem == NestsSelector.DYNAMIC:
            self._run_dynamic_problem(agent_info)

    def _generate_traffic_data(self) -> List[Ride]:
        data_type = self.io.get_user_discrete_choice(GET_DATA_PROMPT,
                                                     [NestsSelector.DEFAULT_DATA,
                                                      NestsSelector.CUSTOM_DATA])

        if data_type == NestsSelector.DEFAULT_DATA:
            return self.traffic_generator.get_default_data()
        elif data_type == NestsSelector.CUSTOM_DATA:
            return self.traffic_generator.get_custom_data()

    def _get_agent_info(self, potential_rides: List[Ride]):

        traffic_simulator: TrafficSimulator = TrafficSimulator(potential_rides)
        incomes_factor: float = self.io.get_user_numerical_choice(
            GET_INCOMES_FACTOR_PROMPT, 0, float("inf"))
        expenses_factor: float = self.io.get_user_numerical_choice(
            GET_EXPENSES_FACTOR_PROMPT, 0, float("inf"))
        incomes_expenses: IncomesExpenses = IncomesExpenses(incomes_factor,
                                                            expenses_factor)
        features_data: FeaturesData = self.features_data_generator. \
            generate_features_data()
        learning_time = int(
            self.io.get_user_numerical_choice(GET_LEARNING_TIME_PROMPT, 0,
                                              float("inf")))
        nests_num: int = int(self.io.get_user_numerical_choice(
            GET_NUMBER_OF_NESTS_PROMPT, MIN_NUMBER_OF_NESTS, MAX_NUMBER_OF_NESTS))
        optional_nests: List[Point] = self. \
            traffic_generator.get_random_nests_locations(nests_num)
        scooters_num: int = int(self.io.get_user_numerical_choice(
            GET_NUMBER_OF_SCOOTER_PROMPT, MIN_NUMBER_OF_SCOOTERS,
            MAX_NUMBER_OF_SCOOTERS))

        return AgentInfo(traffic_simulator, incomes_expenses, features_data,
                         learning_time, optional_nests, scooters_num)

    def _run_static_problem(self, agent_info: AgentInfo):
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_static_agent_legal_values())
        agent: StaticAgent = AgentsFactory.build_static_agent(agent_chosen,
                                                              agent_info)

    def _run_dynamic_problem(self, agent_info: AgentInfo):
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_dynamic_agent_legal_values())
        agent: DynamicAgent = AgentsFactory.build_dynamic_agent(agent_chosen,
                                                                agent_info)
        pass


if __name__ == '__main__':
    a: List[List] = []
    print(a)
