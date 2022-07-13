from programio.consoleio import ConsoleIO
from agents.staticagent import StaticAgent
from agents.dynamicagent import DynamicAgent
from agents.agentsfactory import AgentsFactory
from data.trafficgenerator import TrafficGenerator
from data.featuresdatagenerator import FeaturesDataGenerator
from typing import List

GET_PROBLEM_PROMPT = "Please type problem to solve from the options below:"
GET_NUMBER_OF_SCOOTER_PROMPT = "Please type number of scooters available:"
GET_INCOMES_FACTOR_PROMPT = "Please type incomes factor:"
GET_EXPENSES_FACTOR_PROMPT = "Please type expenses factor:"
GET_LEARNING_TIME_PROMPT = "Please type learning time (seconds):"
GET_AGENT_PROMPT = "Please type agent type:"
GET_DATA_PROMPT = "Please type data choice:"
MIN_NUMBER_OF_SCOOTERS = 1
MAX_NUMBER_OF_SCOOTERS = float("inf")


class ProblemGeneralData:
    def __init__(self, traffic_data: List[List], n: int, incomes_factor: float,
                 expenses_factor: float, learning_time: int):
        self.traffic_data: List[List] = traffic_data  # list of [start time, origin, destination]
        self.n: int = n  # number of optional nests
        self.incomes_factor: float = incomes_factor  #  incomes factor
        self.expenses_factor: float = expenses_factor  #  expenses factor
        self.learning_time: int = learning_time  # number of seconds to run


class NestsSelector:
    DYNAMIC = "dynamic"
    STATIC = "static"
    CUSTOM_DATA = "custom"
    DEFAULT_DATA = "default"

    def __init__(self):
        self.io = ConsoleIO()  # todo - replace with factory for AbstractIO (user chooses which type)
        self.traffic_generator = TrafficGenerator(self.io)
        self.features_data_generator = FeaturesDataGenerator(self.io)

    def run(self):
        problem_data: ProblemGeneralData = self._get_general_data()
        problem = self.io.get_user_discrete_choice(GET_PROBLEM_PROMPT,
                                                   [NestsSelector.DYNAMIC,
                                                    NestsSelector.STATIC])

        if problem == NestsSelector.STATIC:
            self._run_static_problem(problem_data)
        elif problem == NestsSelector.DYNAMIC:
            self._run_dynamic_problem(problem_data)

    def _get_general_data(self) -> ProblemGeneralData:
        traffic_data: List[List] = self._generate_traffic_data()
        n: int = int(self.io.get_user_numerical_choice(GET_NUMBER_OF_SCOOTER_PROMPT,
                                                       MIN_NUMBER_OF_SCOOTERS,
                                                       MAX_NUMBER_OF_SCOOTERS))
        incomes_factor: float = self.io.get_user_numerical_choice(
            GET_INCOMES_FACTOR_PROMPT, 0, float("inf"))
        expenses_factor: float = self.io.get_user_numerical_choice(
            GET_EXPENSES_FACTOR_PROMPT, 0, float("inf"))
        learning_time = int(self.io.get_user_numerical_choice(GET_LEARNING_TIME_PROMPT,
                                                              0, float("inf")))
        return ProblemGeneralData(traffic_data, n, incomes_factor, expenses_factor,
                                  learning_time)

    def _generate_traffic_data(self) -> List[List]:
        data_type = self.io.get_user_discrete_choice(GET_DATA_PROMPT,
                                                     [NestsSelector.DEFAULT_DATA,
                                                      NestsSelector.CUSTOM_DATA])

        if data_type == NestsSelector.DEFAULT_DATA:
            return self.traffic_generator.get_default_data()
        elif data_type == NestsSelector.CUSTOM_DATA:
            return self.traffic_generator.get_custom_data()

    def _run_static_problem(self, problem_data: ProblemGeneralData):
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_static_agent_legal_values())
        agent: StaticAgent = AgentsFactory.build_static_agent(agent_chosen)

        pass

    def _run_dynamic_problem(self, problem_data: ProblemGeneralData):
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_dynamic_agent_legal_values())
        agent: DynamicAgent = AgentsFactory.build_dynamic_agent(agent_chosen)
        pass

    def _generate_default_data(self) -> List[List]:
        """
        proposes to the user few types of default traffic data (by complexity level)
        :return:
        """
        pass

    def _generate_custom_data(self) -> List[List]:
        """

        :return:
        """
        pass


if __name__ == '__main__':
    a: List[List] = []
    print(a)
