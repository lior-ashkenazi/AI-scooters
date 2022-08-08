import numpy as np

from agents.staticagent import StaticAgent
from agents.agent import AgentInfo

from data.trafficdatatypes import *
from data.trafficgenerator import TrafficGenerator

from typing import Tuple

import itertools as it


class GeneticAlgorithmAgent(StaticAgent):
    BASE_NUMBER = 2
    INITIAL_POPULATION_SIZE: int = 2000
    SIMULATION_DAYS_NUM = 10
    SELECTION_PERCENTILE = 80
    DECAY_FACTOR = 0.9
    MUTATION_FACTOR = 0.8
    NOISE_FACTOR = 0.5

    def __init__(self, agent_info: AgentInfo):
        super(GeneticAlgorithmAgent, self).__init__(agent_info)
        self._initial_scooters_locations: Map = \
            TrafficGenerator.get_random_end_day_scooters_locations(self.agent_info.scooters_num)
        self._pop: np.ndarray = self._generate_population(
            GeneticAlgorithmAgent.INITIAL_POPULATION_SIZE)

    def spread_scooters(self) -> Tuple[List[NestAllocation], float]:
        while self._pop.shape[0] > 1:
            fitness_vals: List[float] = []
            for ind in self._pop:
                fitness_vals.append(self._simulate_individual(ind))
            parents: np.ndarray = self._fit(fitness_vals)
            offspring: np.ndarray = self._crossover(parents)
            self._pop = self._mutate(offspring)
            self._pop = self._pop.astype(np.int64)
            print(self._pop.shape)
        # the population size is 1
        the_chosen_one = self._pop[0]
        return self._get_nests_spread(the_chosen_one), self._simulate_individual(the_chosen_one)

    def _fit(self, fitness_vals):
        return self._pop[fitness_vals >= np.percentile(fitness_vals,
                                                       GeneticAlgorithmAgent.SELECTION_PERCENTILE)]

    def _crossover(self, parents):
        offspring = np.empty((0, len(self.agent_info.optional_nests)))
        offspring_size = int(np.max([np.floor(
            self._pop.shape[0] * GeneticAlgorithmAgent.DECAY_FACTOR), 1]))

        # note that a parent can survive a generation and continue to the next one as if he
        # reproduces with itself, he is born anew
        while offspring.shape[0] < offspring_size:
            parent1, parent2 = parents[np.random.choice(parents.shape[0], size=2), :]
            new_offspring = self._reproduce(parent1, parent2)
            offspring = np.vstack([offspring, new_offspring])
            np.unique(offspring, axis=0)
        return np.array(offspring)

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        # in case that offspring.shape[0] == 1 then we won't mutate
        if offspring.shape[0] == 1:
            return offspring
        indices = np.random.choice(offspring.shape[0],
                                   size=int(np.floor(GeneticAlgorithmAgent.MUTATION_FACTOR *
                                                     offspring.shape[0])),
                                   replace=False)
        offspring[indices] = np.apply_along_axis(self._noise, 1, offspring[indices])
        return offspring

    def _generate_population(self, pop_size: int) -> np.ndarray:
        possible_scooters_spread = np.array([comb for comb in
                                             it.product(
                                                 range(self.agent_info.scooters_num + 1),
                                                 repeat=len(self.agent_info.optional_nests))
                                             if sum(comb) == self.agent_info.scooters_num])
        if possible_scooters_spread.shape[0] <= pop_size:
            return possible_scooters_spread
        return possible_scooters_spread[np.random.choice(possible_scooters_spread.shape[0],
                                                         size=pop_size, replace=False), :]

    def _get_nests_spread(self, ind: np.ndarray) -> List[NestAllocation]:
        # we assume that scooters_locations is an action and not a Map type
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(ind)]

    def _simulate_individual(self, ind) -> Union[np.ndarray, float]:
        total_revenue: List[float] = []
        prev_scooters_locations: Map = self._initial_scooters_locations
        ind_nests_spread: List[NestAllocation] = self._get_nests_spread(ind)
        ind_nests_locations: Map = self.agent_info.traffic_simulator. \
            get_scooters_location_from_nests_spread(ind_nests_spread)
        for day in range(GeneticAlgorithmAgent.SIMULATION_DAYS_NUM):
            # simulate
            result: Tuple[List[Ride], Map] = self.agent_info. \
                traffic_simulator.get_simulation_result(ind_nests_locations)
            rides_completed: List[Ride] = result[0]
            next_day_locations: Map = result[1]

            # compute revenue
            cur_revenue: float = self.agent_info.incomes_expenses.calculate_revenue(
                rides_completed, prev_scooters_locations, ind_nests_locations)
            total_revenue.append(cur_revenue)

            prev_scooters_locations = next_day_locations
        return np.mean(total_revenue)

    def _reproduce(self, parent1, parent2) -> np.ndarray:
        offspring = np.zeros(len(self.agent_info.optional_nests))
        matched_parents = np.vstack([parent1, parent2])
        while np.sum(offspring) < self.agent_info.scooters_num:
            parent_idx = np.random.choice(matched_parents.shape[0])
            entry_idx = np.random.choice(np.where(matched_parents[parent_idx] > 0)[0])
            offspring[entry_idx] += 1
            matched_parents[parent_idx][entry_idx] -= 1
        return offspring

    def _noise(self, offspring: np.ndarray) -> np.ndarray:
        scooters_to_displace = round(GeneticAlgorithmAgent.NOISE_FACTOR *
                                     self.agent_info.scooters_num)
        for _ in range(scooters_to_displace):
            valid_indices = np.where(offspring > 0)[0]
            replace = valid_indices.size == 1
            from_entry_idx, to_entry_idx = np.random.choice(np.where(offspring > 0)[0], 2,
                                                            replace=replace)
            offspring[to_entry_idx] += 1
            offspring[from_entry_idx] -= 1
        return offspring
