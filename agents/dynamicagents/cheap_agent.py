import numpy as np
from typing import List, Tuple
from data.trafficdatatypes import Map, NestAllocation, Ride
from data.trafficgenerator import TrafficGenerator

class CheapAgent():
    def __init__(self, env_agent_info, alpha=0.5):
        self.alpha = alpha
        self.agent_info = env_agent_info
        n_actions = len(self.agent_info.optional_nests)
        self.avg = np.ones((n_actions,)) / n_actions

    def get_action(self) -> np.ndarray:
        return self.avg

    def get_start_state(self, normalize=True) -> (Map, np.ndarray):
        end_day_scooters_locations: Map = TrafficGenerator.get_random_end_day_scooters_locations(self.agent_info.scooters_num)
        return end_day_scooters_locations
        # return end_day_scooters_locations, state

    def learn(self, num_games, game_len, visualize):
        best_score = float('-inf')
        score_history = []
        for i in range(num_games):
            scooters_locations: Map
            scooters_locations = self.get_start_state()

            total_rides: List[List[Ride]] = []
            total_nest_allocations: List[List[NestAllocation]] = []
            total_rewards: List[float] = []

            score = 0

            for step_idx in range(game_len):
                action: np.ndarray = self.get_action()
                pre_nests_spread: List[NestAllocation]
                next_day_scooters_locations: Map
                next_state: np.ndarray
                reward: float
                rides_completed: List[Ride]
                pre_nests_spread, next_day_scooters_locations, reward, rides_completed, used_scooters = \
                    self.perform_step(scooters_locations, action)

                self.learn_avg(next_day_scooters_locations)

                total_rides.append(rides_completed)
                total_rewards.append(reward)
                total_nest_allocations.append(pre_nests_spread)
                score += reward
                scooters_locations = next_day_scooters_locations

            # if visualize and (i == num_games - 1):
            #     vis = Visualizer(rides_list=total_rides,
            #                      nests_list=total_nest_allocations,
            #                      revenue_list=total_rewards)
            #     vis.visualise()

            score /= game_len
            score_history.append(score)
            avg_score = np.mean(score_history[-7:])

            # if avg_score > best_score:
            #     best_score = avg_score
            #     if not load_checkpoint:
            #         self.save_models()
            print(action)
            print(np.max(action))
            print(np.argmax(action))
            print('episode ', i, 'score %.5f' % score, 'avg score %.5f' % avg_score)
        return

    def perform_step(self, prev_scooters_locations: Map, action: np.ndarray) -> (Map, np.ndarray, float, List[Ride]):
        prev_nests_spread: List[NestAllocation] = self.get_nests_spread(action)
        prev_nests_locations: Map = self.agent_info.traffic_simulator. \
            get_scooters_location_from_nests_spread(prev_nests_spread)

        # get simulation results - rides completed and scooters final location:
        result: Tuple[List[Ride], Map, Map, List] = self.agent_info. \
            traffic_simulator.get_simulation_result(prev_nests_locations)
        rides_completed: List[Ride] = result[0]
        next_day_locations: Map = result[1]
        potential_starts: Map = result[2]
        used_scooters: List = result[3]

        # compute revenue
        reward: float = self.agent_info.incomes_expenses.calculate_revenue(
            rides_completed, prev_scooters_locations, prev_nests_locations)
        return prev_nests_spread, next_day_locations, reward, rides_completed, used_scooters

    def learn_avg(self, next_day_scooters_locations):
        points = next_day_scooters_locations.get_points()
        nests = np.array([(n.x, n.y) for n in self.agent_info.optional_nests])
        dists = np.sum((nests[None,:,:] - points[:, None, :])**2, axis=-1)
        bin_by_closest = np.bincount(np.argmin(dists, axis=-1), minlength=len(self.avg))
        new_avg = bin_by_closest / bin_by_closest.sum()
        self.avg = (1 - self.alpha) * self.avg + self.alpha * new_avg

    def get_nests_spread(self, action: np.ndarray) -> List[NestAllocation]:
        # we assume that scooters_locations is an action and not a Map type
        # discretize the action-
        discrete_action = self.discretize(action)
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(discrete_action)]

    def discretize(self, data: np.ndarray) -> np.ndarray:
        data_copy = data.copy()
        assert np.isclose(np.sum(data_copy), 1)
        data_copy *= self.agent_info.scooters_num
        data_fraction, data_int = np.modf(data_copy)

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
