import sys

from programio.abstractio import AbstractIO
from programio.consoleio import ConsoleIO
from programio.graphicalio import GraphicIO

from agents.agent import AgentInfo
from agents.staticagent import StaticAgent
from agents.dynamicagent import DynamicAgent
from agents.agentsfactory import AgentsFactory

from data.trafficgenerator import TrafficGenerator
from data.trafficdatatypes import *
from data.featuresdatagenerator import FeaturesDataGenerator, FeaturesData

from simulation.trafficsimulator import TrafficSimulator
from simulation.incomesexpenses import IncomesExpenses

from typing import List, Tuple

import argparse

import json

GET_PROBLEM_PROMPT = "Please type problem to solve from the options below:"
GET_NUMBER_OF_SCOOTER_PROMPT = "Please type number of scooters available:"
GET_NUMBER_OF_NESTS_PROMPT = "Please type number of optional nests:"
GET_INCOMES_FACTOR_PROMPT = "Please type incomes factor:"
GET_EXPENSES_FACTOR_PROMPT = "Please type expenses factor:"
GET_LEARNING_TIME_PROMPT = "Please type learning time (seconds):"
GET_ITERATIONS_NUMBER_DYNAMIC_RUN_PROMPT = "Please enter number of iterations to run" \
                                           " in the end of the learning process:"
GET_DYNAMIC_RL_GRID_LENGTH = "Please enter state grid length:"
GET_RL_EPSILON = "Please enter an epsilon:"
GET_AGENT_PROMPT = "Please type agent type:"
GET_DATA_PROMPT = "Please type data choice:"
MIN_NUMBER_OF_SCOOTERS = 1
MAX_NUMBER_OF_SCOOTERS = float("inf")
MIN_NUMBER_OF_NESTS = 1
MAX_NUMBER_OF_NESTS = float("inf")
MIN_DYNAMIC_RL_GRID_LENGTH = 1
MAX_DYNAMIC_RL_GRID_LENGTH = 1000
MIN_RL_EPSILON = 0
MAX_RL_EPSILON = 1
SEARCH_RADIUS = 0.2  # in km, todo - get this value as an input? what value to assign?


# this value should represent the radius in terms of euclidean
#  distance in x,y coordinates of the map (not meters)


class NestsSelector:
    DYNAMIC_AGENT = ["dynamic", "d"]
    STATIC_AGENT = ["static", "s"]
    CUSTOM_DATA = ["custom", "c"]
    DEFAULT_DATA = ["default", "d"]
    CONSOLE_IO = ["console", "c"]
    GRAPHIC_IO = ["graphic", "g"]

    def __init__(self, io: AbstractIO = ConsoleIO()):
        self.io = io
        self.traffic_generator: TrafficGenerator = TrafficGenerator(self.io)
        self.features_data_generator: FeaturesDataGenerator = FeaturesDataGenerator(
            self.io)

    def run(self):
        potential_rides: List[Ride] = self._generate_traffic_data()
        agent_info: AgentInfo = self._get_agent_info(potential_rides)

        if args.agent in NestsSelector.STATIC_AGENT:
            self._run_static_problem(agent_info)
        elif args.agent in NestsSelector.DYNAMIC_AGENT:
            iterations_num: int = int(
                self.io.get_user_numerical_choice(
                    GET_ITERATIONS_NUMBER_DYNAMIC_RUN_PROMPT, 0, float("inf")))
            self._run_dynamic_problem(agent_info, iterations_num)

    def _generate_traffic_data(self) -> List[Ride]:
        # return the data requested:
        if args.data in NestsSelector.DEFAULT_DATA:
            return self.traffic_generator.get_default_data()
        elif args.data in NestsSelector.CUSTOM_DATA:
            return self.traffic_generator.get_custom_data()

    def _get_agent_info(self, potential_rides: List[Ride]):

        traffic_simulator: TrafficSimulator = TrafficSimulator(potential_rides,
                                                               SEARCH_RADIUS)
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
            traffic_generator.get_optional_nests_locations(nests_num)
        scooters_num: int = int(self.io.get_user_numerical_choice(
            GET_NUMBER_OF_SCOOTER_PROMPT, MIN_NUMBER_OF_SCOOTERS,
            MAX_NUMBER_OF_SCOOTERS))

        return AgentInfo(traffic_simulator, incomes_expenses, features_data,
                         learning_time, optional_nests, scooters_num)

    def _run_static_problem(self, agent_info: AgentInfo):

        # get agent:
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_static_agent_legal_values())
        agent: StaticAgent = AgentsFactory.build_static_agent(agent_chosen, agent_info)

        # get result:
        result: Tuple[List[NestAllocation], float] = agent.spread_scooters()
        spread_points: List[NestAllocation] = result[0]
        revenue: float = result[1]

        # show result
        self._show_static_results(agent, spread_points, revenue)

    def _run_dynamic_problem(self, agent_info: AgentInfo, iterations_num: int):

        # get agent:
        agent_chosen: str = self.io.get_user_discrete_choice(
            GET_AGENT_PROMPT, AgentsFactory.get_dynamic_agent_legal_values())
        if agent_chosen == AgentsFactory.AGENT_DYNAMIC_RL:
            agent_info.epsilon = \
                self.io.get_user_numerical_choice(GET_RL_EPSILON,
                                                  MIN_RL_EPSILON,
                                                  MAX_RL_EPSILON)
            agent_info.grid_len = \
                int(self.io.get_user_numerical_choice(GET_DYNAMIC_RL_GRID_LENGTH,
                                                      MIN_DYNAMIC_RL_GRID_LENGTH,
                                                      MAX_DYNAMIC_RL_GRID_LENGTH))
        agent: DynamicAgent = AgentsFactory.build_dynamic_agent(agent_chosen, agent_info)

        # learn:
        agent.learn()

        # show result:
        avg_revenue: float = agent.get_average_revenue(iterations_num)
        self._show_dynamic_results(agent, avg_revenue)

    def run_demo(self, args):
        with open('consts.json', 'r') as f:
            consts = json.load(f)

        traffic_simulator: TrafficSimulator = TrafficSimulator(consts['samples_num'],
                                                               SEARCH_RADIUS,
                                                               const_rides=args.const_rides)
        incomes_factor: float = consts['incomes_factor']
        expenses_factor: float = consts['expenses_factor']
        incomes_expenses: IncomesExpenses = IncomesExpenses(incomes_factor,
                                                            expenses_factor)
        features_data: FeaturesData = self.features_data_generator. \
            generate_features_data()
        learning_time = consts['learning_time']
        optional_nests: List[Point] = self.traffic_generator. \
            get_not_random_nests_locations(consts['optional_nests'])
        scooters_num: int = consts['scooters_num']
        epsilon = consts['epsilon']
        grid_len = consts['grid_len']

        agent_info = AgentInfo(traffic_simulator, incomes_expenses, features_data, learning_time,
                               optional_nests, scooters_num, epsilon, grid_len)

        # agent_chosen = consts['agent_chosen']
        agent_chosen = args.agent_chosen
        if agent_chosen in ['dynamic_RL', 'baseline_agent']:
            agent: DynamicAgent = AgentsFactory.build_dynamic_agent(agent_chosen, agent_info)
            agent.learn(num_games=args.num_games, game_len=args.game_len, visualize=True,
                        pretrain_critic=args.pretrain_critic, load_checkpoint=args.load_critic)
        elif agent_chosen == 'genetic_algorithm':
            agent: StaticAgent = AgentsFactory.build_static_agent(agent_chosen, agent_info)
            agent.spread_scooters()

        # agent_chosen = "genetic_algorithm"
        #
        # agent: StaticAgent = AgentsFactory.build_static_agent(agent_chosen, agent_info)
        #
        # spread_points, revenue = agent.spread_scooters()
        #
        # self._show_static_results(agent, spread_points, revenue)

    def _show_static_results(self, agent: StaticAgent,
                             spread_points: List[NestAllocation],
                             revenue: float):
        self.io.show_value("revenue:", revenue)
        self.io.confirm_and_continue()
        self.io.show_spread(spread_points)
        # todo - after building the simulator, think of a way to show a simulation,
        #        maybe in the simulator use an option of simulation for visualization
        #        and use "yield" to get stated during the simulation
        # start_points: List[Point] = agent.agent_info.traffic_simulator.\
        #     get_scooters_location_from_nests_spread(spread_points)

    def _show_dynamic_results(self, agent: DynamicAgent, avg_revenue: float):
        self.io.show_value("average revenue:", avg_revenue)
        self.io.confirm_and_continue()
        # todo - think of how to present the result of the dynamic agent after
        # building the simulator
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--io",
                        default='c',
                        help="The input/output interface of the program",
                        choices=NestsSelector.CONSOLE_IO + NestsSelector.GRAPHIC_IO)
    parser.add_argument("-a",
                        "--agent",
                        default='d',
                        help="The type of agent",
                        choices=NestsSelector.DYNAMIC_AGENT + NestsSelector.STATIC_AGENT)
    parser.add_argument("-d",
                        "--data",
                        default='d',
                        help="The type of data",
                        choices=NestsSelector.DEFAULT_DATA + NestsSelector.CUSTOM_DATA)
    parser.add_argument("--const_rides", action="store_true",
                        help="if true, it will use the same potential rides for each day")
    parser.add_argument("--pretrain_critic", action="store_true")
    parser.add_argument("--load_critic", action="store_true")
    parser.add_argument("--num_games", action="store", type=int,
                        help="number of games, each one starts with a random location")
    parser.add_argument("--game_len", action="store", type=int,
                        help="number of days in a game")
    parser.add_argument("--agent_chosen", action="store", type=str)
    args = parser.parse_args()
    # TODO to be deleted in the future
    # if len(sys.argv) == 1:
    ns = NestsSelector()
    ns.run_demo(args)
    # else:
    #     io = GraphicIO() if args.io in NestsSelector.GRAPHIC_IO else ConsoleIO()
    #     ns = NestsSelector(io)
    #     ns.run()
