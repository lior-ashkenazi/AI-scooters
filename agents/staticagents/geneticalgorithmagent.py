import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl

from agents.staticagent import StaticAgent
from agents.agent import AgentInfo

from data.trafficdatatypes import *
from data.trafficgenerator import TrafficGenerator

from typing import Tuple
from scipy.special import softmax

import itertools as it


class GeneticAlgorithmAgent(StaticAgent):
    BASE_NUMBER = 2
    INITIAL_POPULATION_SIZE: int = 500
    SIMULATION_DAYS_NUM = 4
    SELECTION_PERCENTILE = 90
    DECAY_FACTOR = 0.9
    MUTATION_FACTOR = 0.8
    NOISE_FACTOR = 0.20
    UNUSED_SCOOTERS_FACTOR = 5

    def __init__(self, agent_info: AgentInfo, model_dir):
        super(GeneticAlgorithmAgent, self).__init__(agent_info)
        self.model_dir = os.path.join('models', model_dir)
        self._initial_scooters_locations: Map = \
            TrafficGenerator.get_random_end_day_scooters_locations(self.agent_info.scooters_num)
        self._pop: np.ndarray = self._generate_population(
            GeneticAlgorithmAgent.INITIAL_POPULATION_SIZE)

    def spread_scooters(self, dump=True) -> Tuple[List[NestAllocation], float]:
        generation = 0
        log = {'generations': {}}
        total_money = []
        while self._pop.shape[0] > 1:
            generation += 1
            print(f"Simulating generation: {generation}, population: {len(self._pop)}")
            fitness_vals, money_vals = [], []
            generation_log = list()
            for ind in self._pop:
                income, expense, unused, nest_spread, rides_completed = self._simulate_individual(ind)
                fitness_vals.append(income - expense - unused)
                money_vals.append(income - expense)
                generation_log.append({'income': income, 'expense': expense, 'unused': unused,
                                       'nest_spread': nest_spread, 'rides_completed': rides_completed})
            log['generations'][generation] = generation_log
            best_fit = np.argmax(fitness_vals)
            print(f'Avg {round(np.average(fitness_vals), 2)} '
                  f'Max fit: {round(fitness_vals[best_fit], 2)}\n'
                  f'Avg money {round(np.average(money_vals), 2)} '
                  f'Max money: {round(money_vals[best_fit], 2)}\n'
                  f'Argmax: {self._pop[np.argmax(fitness_vals)]}')
            total_money.append(money_vals[best_fit])
            parents: np.ndarray = self._fit(fitness_vals)
            offspring: np.ndarray = self._crossover(parents)
            offspring = self._mutate(offspring)
            self._pop = offspring
        # the population size is 1
        the_chosen_one = self._pop[0]
        if dump:
            log['total_money'] = total_money
            log['nest_locations'] = self.agent_info.optional_nests
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)
            with open(os.path.join(self.model_dir, 'genetic.pkl'), 'wb') as f:
                pkl.dump(log, f)
            with open(os.path.join(self.model_dir, 'genetic.pkl'), 'rb') as f:
                a = pkl.load(f)
        return self._get_nests_spread(the_chosen_one), self._simulate_individual(the_chosen_one)

    def _fit(self, fitness_vals):
        return self._pop[fitness_vals >= np.percentile(fitness_vals,
                                                       GeneticAlgorithmAgent.SELECTION_PERCENTILE)]

    def _crossover(self, parents):
        if parents.shape[0] == 1:
            return parents
        offspring = list()
        offspring_size = int(np.max([np.floor(
            self._pop.shape[0] * GeneticAlgorithmAgent.DECAY_FACTOR), 1]))

        # note that a parent can survive a generation and continue to the next one as if he
        # reproduces with itself, he is born anew
        for i in range(offspring_size):
            parent1, parent2 = parents[np.random.choice(parents.shape[0], size=2, replace=False), :]
            offspring.append(self._reproduce(parent1, parent2))
        return np.array(offspring)

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        # in case that offspring.shape[0] == 1 then we won't mutate
        if offspring.shape[0] == 1:
            return offspring
        indices = np.random.choice(offspring.shape[0],
                                   size=int(np.floor(GeneticAlgorithmAgent.MUTATION_FACTOR *
                                                     offspring.shape[0])),
                                   replace=False)
        offspring[indices] = self._noise(offspring[indices])
        return offspring

    def _generate_population(self, pop_size: int) -> np.ndarray:
        population = np.power(np.random.random((pop_size, len(self.agent_info.optional_nests))), 30)
        population /= np.expand_dims(np.sum(population, axis=1), axis=-1)
        return population


    def _get_nests_spread(self, ind: np.ndarray) -> List[NestAllocation]:
        scooter_spread = self.discretize(ind)
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(scooter_spread)]

    def _simulate_individual(self, ind):
        income : List[float] = []
        expense : List[float] = []
        unused : List[float] = []
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
            unused_scooters = (self.agent_info.scooters_num - len(rides_completed)) * self.UNUSED_SCOOTERS_FACTOR
            # compute revenue
            cur_revenue = self.agent_info.incomes_expenses.calculate_revenue(
                rides_completed, prev_scooters_locations, ind_nests_locations)
            income.append(cur_revenue[0])
            expense.append(cur_revenue[1])
            unused.append(unused_scooters)
            prev_scooters_locations = next_day_locations
        return np.mean(income), np.mean(expense), np.mean(unused), ind_nests_spread, rides_completed

    def _reproduce(self, parent1, parent2) -> np.ndarray:
        # todo attempt 1: half of coordinates from each parent
        # take_from_p1 = np.random.choice(np.arange(len(parent1)), len(parent1) // 2, replace=False)
        take_from_p1 = np.random.random(len(parent1)) > 0.5
        offspring = take_from_p1 * parent1 + (1 - take_from_p1) * parent2
        offspring = offspring / offspring.sum()
        return offspring
        # todo attempt 2: average of vectors then normalize

    def _noise(self, offspring: np.ndarray) -> np.ndarray:
        n_offsprings, size_offspring = offspring.shape
        noisy_offspring = offspring.copy()

        for i in range(n_offsprings // 4):
            # add noise in 1 random coordinate for each offspring
            noise = np.random.normal(scale=GeneticAlgorithmAgent.NOISE_FACTOR,
                                     size=n_offsprings)
            coords = np.random.randint(low=0, high=size_offspring, size=n_offsprings)
            noisy_offspring[:, coords] += np.clip(noisy_offspring[:, coords] + noise, a_min=0, a_max=np.inf)
        # normalize
        noisy_offspring = noisy_offspring / noisy_offspring.sum(axis=-1)[:, None]
        return noisy_offspring

    def discretize(self, action_data: np.ndarray) -> np.ndarray:
        data = action_data.copy()
        assert np.isclose(np.sum(data), 1)
        data *= self.agent_info.scooters_num
        data_fraction, data_int = np.modf(data)

        round_up_amount = np.sum(data_fraction)
        # assert np.isclose(round_up_amount, round(round_up_amount)), f"{data}"
        round_up_amount = round(round_up_amount)

        data_fraction_flat, data_int_flat = data_fraction.flatten(), data_int.flatten()
        data_fraction_index = np.argsort(-data_fraction_flat)
        data_int_flat[data_fraction_index[:round_up_amount]] += 1
        data_discrete = data_int_flat.reshape(data.shape)
        assert np.sum(data_discrete) == self.agent_info.scooters_num
        data_discrete = data_discrete.astype(int)
        return data_discrete