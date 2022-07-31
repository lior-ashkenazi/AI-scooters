import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

from agents.agent import AgentInfo
from agents.dynamicagent import DynamicAgent
from agents.rlagent import ReinforcementLearningAgent

from data.trafficdatatypes import *
from data.trafficgenerator import TrafficGenerator

from data.trafficdatatypes import Map, NestAllocation

from typing import Tuple

import random
import itertools as it


class DynamicRLAgent(ReinforcementLearningAgent, DynamicAgent):
    def learn(self):
        prev_scooters_locations: Map = TrafficGenerator.get_random_end_day_scooters_locations(
            self.agent_info.scooters_num)
        state = self.get_state(prev_scooters_locations)
        rides_completed = []
        total_revenue = []
        states = [state]
        actions = []
        # TODO - how do we iterate the process of learning?
        for i in range(10):
            # choice = random.choice([self.epsilon, 1 - self.epsilon])
            # if choice == self.epsilon:
            #     action = self.get_action()
            # elif choice == 1 - self.epsilon:
            #     action = self.q_func(state) or self.get_action(state)
            action: np.ndarray = self.get_action()

            # get the destination map
            prev_nests_spread: List[NestAllocation] = self.get_nests_spread(action)
            prev_nests_locations: Map = self.agent_info.traffic_simulator. \
                get_scooters_location_from_nests_spread(prev_nests_spread)

            # get simulation results - rides completed and scooters final location:
            result: Tuple[List[Ride], Map] = self.agent_info. \
                traffic_simulator.get_simulation_result(prev_nests_locations)
            rides_completed: List[Ride] = result[0]
            next_day_locations: Map = result[1]

            # compute revenue
            cur_revenue: float = self.agent_info.incomes_expenses.calculate_revenue(
                rides_completed, prev_scooters_locations, prev_nests_locations)
            total_revenue.append(cur_revenue)

            # TODO NN can learn here after calculating revenue

            prev_scooters_locations = next_day_locations
            state = self.get_state(prev_scooters_locations)
            states.append(state)
            actions.append(action)
            print(f'finished {i}th iteration')
        plt.scatter(range(10), total_revenue)
        plt.show()

    def get_state(self, end_day_scooters_locations: Map) -> np.ndarray:
        binx: np.ndarray
        biny: np.ndarray
        binx, biny = TrafficGenerator.get_coordinates_bins(self.agent_info.grid_len)
        state = binned_statistic_2d(end_day_scooters_locations[:, 0],
                                    end_day_scooters_locations[:, 1],
                                    None, 'count', bins=[binx, biny]).statistic
        assert end_day_scooters_locations.get_points().shape[0] == int(np.sum(state))
        return state

    def get_action(self, *args) -> np.ndarray:
        # TODO - maybe this method is where we want to implement Deep RL, so the current content
        #  of this method can move to another method or stay here and add to args a state
        if not args:
            # extremely inefficient, do not use often
            possible_scooters_spread = [comb for comb in
                                        it.combinations_with_replacement(
                                            range(self.agent_info.scooters_num + 1),
                                            len(self.agent_info.optional_nests))
                                        if sum(comb) == self.agent_info.scooters_num]
            return np.array(random.choice(possible_scooters_spread))

    def get_nests_spread(self, action: np.ndarray) -> List[NestAllocation]:
        # we assume that scooters_locations is an action and not a Map type
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(action)]
