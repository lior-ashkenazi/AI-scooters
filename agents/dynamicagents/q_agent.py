import numpy as np
from typing import List, Tuple
from data.trafficdatatypes import Map, NestAllocation, Ride
from data.trafficgenerator import TrafficGenerator
from agents.agent import AgentInfo
class State:
    def __init__(self, starting_points, scooter_spread, scooters_per_nest, unused_scooters):
        self.starting_points = starting_points
        self.scooter_spread = scooter_spread
        self.scooters_per_nest = scooters_per_nest
        self.unused_scooters = unused_scooters

ACTION_PERCENTILES = [0.2 * i for i in range(5)]
class FeatureExtractor:
    def __init__(self, agent_info):
        self.agent_info = agent_info
        self.nest_bins = np.array([(n.x, n.y) for n in self.agent_info.optional_nests])
        self.n_actions = len(self.agent_info.optional_nests)
        self.action_vec_shape = (self.n_actions ** 2 * len(ACTION_PERCENTILES)) + 1
        self.state_vec_shape = self.n_actions * 3
        self.feature_shape = self.state_vec_shape + self.action_vec_shape
        self.scooters_num = agent_info.scooters_num

    def __call__(self, state, action):
        state_vec = self.state_to_vec(state)
        action_to_vec = self.action_to_vec(action)
        return np.concatenate([state_vec, action_to_vec])

    def state_to_vec(self, state):
        binx: np.ndarray
        biny: np.ndarray
        endpoints = state.scooter_spread
        startpoints = state.starting_points

        if len(endpoints.get_points()) == 0:
            end_dist_counts = np.zeros((self.n_actions))
        else:
            end_dist = np.sum(110 * (endpoints[None, ...] - self.nest_bins[:, None, :]) **2, axis=-1)
            end_dist_counts = np.bincount(end_dist.argmin(axis=0), minlength=self.n_actions)
            end_dist_counts = end_dist_counts / end_dist_counts.sum()

        # self.avg_end = (1 - self.alpha) * self.avg_end + self.alpha * end_dist_counts
        # cur_end_means = end_dist.mean(axis=1)

        if len(startpoints.get_points()) == 0:
            start_dist_counts = np.zeros((self.n_actions,))
        else:
            start_dist = np.sum(110 * (startpoints[None, ...] - self.nest_bins[:, None, :]) **2, axis=-1)
            start_dist_counts = np.bincount(start_dist.argmin(axis=0), minlength=self.n_actions)
            start_dist_counts = start_dist_counts / start_dist_counts.sum()
        # cur_start_means = start_dist.mean(axis=1)

        scooters_per_nest = state.scooters_per_nest / state.scooters_per_nest.sum()
        return np.concatenate([end_dist_counts,
                               start_dist_counts,
                               scooters_per_nest])

    # def state_to_vec(self, state):
    #
    #     endpoints = state.scooter_spread
    #     end_dist = np.sum(110 * (endpoints[None, ...] - self.nest_bins[:, None, :]) **2, axis=-1)
    #     end_dist_counts = np.bincount(end_dist.argmin(axis=0), minlength=self.n_actions)
    #     end_dist_counts = end_dist_counts / end_dist_counts.sum()
    #     scooters_per_nest = state.scooters_per_nest / state.scooters_per_nest.sum()
    #     return np.concatenate([end_dist_counts, scooters_per_nest])

    def action_to_vec(self, action):
        v = np.zeros(self.action_vec_shape)
        if action == 'nop':
            v[-1] = 1
        else:
            index = np.ravel_multi_index(action, dims=(self.n_actions, self.n_actions, len(ACTION_PERCENTILES)))
            v[index] = 1
        return v




class Qagent():

    def __init__(self,env_agent_info : AgentInfo,
                 epsilon = 0.3, alpha = 1e-1, gamma=1e-1):

        self.agent_info = env_agent_info
        self.feature_extractor = FeatureExtractor(env_agent_info)
        self.n_actions = len(self.agent_info.optional_nests)
        self.scooters_num = self.agent_info.scooters_num
        self.qvalues = dict()

        self.w = np.zeros(self.feature_extractor.feature_shape)

        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)


    def get_legal_actions(self, state: State):
        # start with nop
        l = {"nop"}

        # add (i, j) if i != j and i has scooters
        for from_nest in np.where(state.scooters_per_nest)[0]:
            for to_nest in range(len(state.scooters_per_nest)):
                if from_nest != to_nest:
                    for amt_idx in range(len(ACTION_PERCENTILES)):
                        l.add((from_nest, to_nest, amt_idx))
        return l

    def getValue(self, state, legal_actions):
        return max(self.getQValue(state, action)
                   for action in legal_actions)

    def getQValue(self, state, action):
        f = self.feature_extractor(state, action)
        return np.dot(self.w, f)

    def get_policy(self, state, legal_actions):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_val = self.getValue(state, legal_actions)
        max_actions = [action for action in legal_actions if
                       self.getQValue(state, action) == max_val]
        i = np.random.randint(0, len(max_actions))
        return max_actions[i]

    def get_action(self, state):
        legal_actions = self.get_legal_actions(state)
        action = None
        if len(legal_actions) > 0:
            if np.random.random() > self.epsilon:
                i = np.random.randint(0, len(legal_actions))
                return list(legal_actions)[i]
            else:
                action = self.get_policy(state, legal_actions)
        return action

    def update(self, state, action, next_state, reward, unused_scooters):
        """
           Should update your weights based on transition
        """
        final_reward = reward[0] - reward[1] - (30 * unused_scooters)

        legal_actions = self.get_legal_actions(state)
        best_legal_action = max(self.getQValue(next_state, action) for action in legal_actions) \
            if len(legal_actions) > 0 else 0
        f = self.feature_extractor(state, action)
        correction = (final_reward + self.discount * best_legal_action) - self.getQValue(state, action)
        self.w = self.w + (self.alpha * correction * f)

    def learn(self, num_games, game_len):
        best_score = float('-inf')
        score_history = []
        for i in range(num_games):
            state: State
            state = self.get_start_state()
            total_rewards: List[float] = []

            score = 0
            for step_idx in range(game_len):
                action = self.get_action(state)
                next_state, reward = self.perform_step(state, action, step_idx % 2)
                # print(state.scooters_per_nest, action, reward[0] - reward[1])
                self.update(state, action, next_state, reward, next_state.unused_scooters)
                total_rewards.append(reward)
                score += (reward[0] - reward[1])
                state = next_state

            score /= game_len
            score_history.append(score)
            avg_score = np.mean(score_history[-7:])
            print('episode ', i, 'score %.5f' % score, 'avg score %.5f' % avg_score)
        return

    def perform_step(self, state, action, options_index):
        old_scooter_spread = state.scooter_spread
        new_nest_count, new_nest_spread = self.get_new_nest_spread(old_nest_spread=state.scooters_per_nest,
                                                   action=action)
        scooters_based_on_new_spread = self.agent_info.traffic_simulator.\
            get_scooters_location_from_nests_spread(new_nest_spread)

        # get simulation results - rides completed and scooters final location:
        result = self.agent_info. \
            traffic_simulator.get_simulation_result(scooters_based_on_new_spread, options_index)
        rides_completed = result[0]
        new_scooter_spread = result[1]
        starting_points = result[2]
        unused_scooters = (self.scooters_num - len(rides_completed)) / self.scooters_num
        reward = self.agent_info.incomes_expenses.calculate_revenue(rides_completed,
                                                                    old_scooter_spread,
                                                                    scooters_based_on_new_spread)
        next_state = State(starting_points=starting_points,
                           scooter_spread=new_scooter_spread, scooters_per_nest=np.array(new_nest_count),
                           unused_scooters=unused_scooters)
        return next_state, reward

    def get_start_state(self):
        nest_spread = discretize(np.ones((self.n_actions,)) / self.n_actions, self.agent_info.scooters_num)
        nests = [NestAllocation(self.agent_info.optional_nests[i], nest_spread[i]) for i in range(len(nest_spread))]
        scooter_spread: Map = self.agent_info.traffic_simulator.\
            get_scooters_location_from_nests_spread(nests)
        return State(scooter_spread, scooter_spread, nest_spread, 0)

    def get_new_nest_spread(self, old_nest_spread, action):
        new_nest_count = [val for val in old_nest_spread]
        if action != 'nop':
            amount_to_add = round(new_nest_count[action[0]] * ACTION_PERCENTILES[action[2]])
            new_nest_count[action[0]] -= amount_to_add
            new_nest_count[action[1]] += amount_to_add
        return new_nest_count, [NestAllocation(self.agent_info.optional_nests[i], new_nest_count[i])
                for i in range(len(new_nest_count))]


def discretize(data, n):
    data_copy = data.copy()
    # assert np.isclose(np.sum(data_copy), 1)
    data_copy *= n
    data_fraction, data_int = np.modf(data_copy)

    round_up_amount = np.sum(data_fraction)
    # assert np.isclose(round_up_amount, round(round_up_amount)), f"{data}"
    round_up_amount = round(round_up_amount)

    data_fraction_flat, data_int_flat = data_fraction.flatten(), data_int.flatten()
    data_fraction_index = np.argsort(-data_fraction_flat)
    data_int_flat[data_fraction_index[:round_up_amount]] += 1
    data_discrete = data_int_flat.reshape(data.shape)
    assert np.sum(data_discrete) == n
    data_discrete = data_discrete.astype(int)
    return data_discrete
