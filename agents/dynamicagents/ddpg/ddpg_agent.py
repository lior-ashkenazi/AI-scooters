import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

import numpy as np
from scipy.stats import binned_statistic_2d
from typing import List, Tuple
import random
import itertools as it

from agents.dynamicagents.ddpg.buffer import ReplayBuffer
from agents.dynamicagents.ddpg.networks import ActorNetwork, CriticNetwork
# from agents.dynamicagents.ddpg.utils import plot_learning_curve

from agents.dynamicagent import DynamicAgent
from agents.rlagent import ReinforcementLearningAgent
from data.trafficdatatypes import Map, NestAllocation, Ride
from data.trafficgenerator import TrafficGenerator
from programio.visualizer import Visualizer
import os


class DdpgAgent(ReinforcementLearningAgent, DynamicAgent):
    def __init__(self, env_agent_info, model_dir, unused_scooters_factor, actor_lr=2e-4, critic_lr=5e-3,
                 decay_factor=0.6, max_size=250, target_update_rate=1e-3,
                 batch_size=64, noise=0.007):
        super(DdpgAgent, self).__init__(env_agent_info)
        self.model_dir = os.path.join(r'C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters\models', model_dir)
        self.unused_scooters_factor = unused_scooters_factor
        self.decay_factor = decay_factor
        self.target_update_rate = target_update_rate
        # input_shape = (self.agent_info.grid_len, self.agent_info.grid_len, 2)
        self.n_actions = len(self.agent_info.optional_nests)
        input_shape = (self.n_actions, 2)
        self.memory = ReplayBuffer(max_size, input_shape, self.n_actions)
        self.random_memory = ReplayBuffer(max_size, input_shape, self.n_actions)
        self.batch_size = batch_size
        self.min_action = 0
        self.max_action = 1
        self.noise = noise
        # self.actor = ActorNetwork(env_agent_info, name='actor')
        self.actor = ActorNetwork(self.n_actions, name='actor', scooters_num=self.agent_info.scooters_num,
                                  rides_num=self.agent_info.traffic_simulator._rides_per_day_part)
        self.critic = CriticNetwork(name='critic', chkpt_dir=self.model_dir)
        # self.target_actor = ActorNetwork(env_agent_info,
        #                                  name='target_actor')
        self.target_actor = ActorNetwork(self.n_actions,
                                         name='target_actor', chkpt_dir=self.model_dir,
                                         scooters_num=self.agent_info.scooters_num,
                                         rides_num=self.agent_info.traffic_simulator._rides_per_day_part)
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        # self.actor.built, self.critic.built, self.target_actor.built, self.target_critic.built = True, True, True, True

        self.update_network_parameters(target_update_rate=1)
        self.nest_bins = np.array([(n.x, n.y) for n in self.agent_info.optional_nests])
        self.avg_start = np.ones((self.n_actions,)) / self.n_actions
        self.alpha = 1

    def update_network_parameters(self, target_update_rate=None):
        if target_update_rate is None:
            target_update_rate = self.target_update_rate

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * target_update_rate + targets[i] * (1 - target_update_rate))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * target_update_rate + targets[i] * (1 - target_update_rate))
        self.target_critic.set_weights(weights)

    def remember(self, memory, state, extra_features, action, reward, next_state):
        memory.store_transition(state, extra_features, action, reward, next_state)

    def save_models(self):
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        print('... saving models ...')
        # self.actor.save_weights(self.actor.checkpoint_file)
        # self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        # self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        # self.actor.load_weights(self.actor.checkpoint_file)
        # self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic(np.zeros((1, self.n_actions, 2)), np.zeros((1, self.n_actions)), np.zeros((1, 1)), np.zeros((1, 1)))
        self.target_critic(np.zeros((1, self.n_actions, 2)), np.zeros((1, self.n_actions)), np.zeros((1, 1)), np.zeros((1, 1)))
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.critic.checkpoint_file)  # todo: change back to target loading!

    def get_action(self, state, evaluate=False):
        # return state[..., 1]
        actor_input = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(actor_input)
        action = action.numpy()
        # print(action, state[..., 1], state[..., 0])
        # print("dist:", np.abs(action - state[..., 1]).mean())
        if not evaluate:
            action += np.random.normal(scale=self.noise, size=self.n_actions)
            # action += tf.random.normal(shape=[self.n_actions],
            #                            mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        action = np.clip(action, self.min_action, self.max_action)[0]
        if np.all(action == 0):
            action = np.full(action.shape, 1)
        action /= np.sum(action)
        return action

    def learn_batch(self, memory, grad_actor, grad_critic):
        if memory.mem_cntr < self.batch_size:
            return 0, 0, [0], [0]

        state, extra_feature, action, reward, next_state = memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        extra_features = tf.convert_to_tensor(extra_feature, dtype=tf.float32)
        actor_loss, critic_loss, critic_value_lst = 0, 0, [0]
        if grad_critic:
            with tf.GradientTape() as tape:
                target_actions = self.target_actor(next_states)
                next_optimal_transport_approx, next_unused_scooters_percent = self.actor.extract_extra_features(states, target_actions)
                critic_value_next = tf.squeeze(self.target_critic(next_states, target_actions, next_optimal_transport_approx, next_unused_scooters_percent), 1)
                optimal_transport_approx, unused_scooters_percent = self.actor.extract_extra_features(states, actions)
                critic_value = tf.squeeze(self.critic(states, actions, optimal_transport_approx, unused_scooters_percent), 1)
                target = rewards + self.decay_factor * critic_value_next
                # target = rewards
                critic_loss = keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss,
                                                    self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(
                critic_network_gradient, self.critic.trainable_variables))
            critic_value_lst = [v for v in critic_value.numpy()]
        if grad_actor:
            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states)
                optimal_transport_approx, unused_scooters_percent = self.actor.extract_extra_features(states, new_policy_actions)
                actor_loss = -self.critic(states, new_policy_actions, optimal_transport_approx, unused_scooters_percent)
                # actor_loss = keras.losses.MSE(new_policy_actions, states[..., 1])
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss,
                                                   self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(
                actor_network_gradient, self.actor.trainable_variables))
        if grad_critic and grad_actor:
            self.update_network_parameters()
        reward_lst = [v for v in rewards.numpy()]
        return critic_loss, actor_loss, critic_value_lst, reward_lst

    def extract_extra_features(self, state, action):
        locations_end_day = state[..., 1]
        potential_rides_start = state[..., 0]

        moving_dist = np.sum(np.abs(locations_end_day - action))
        total_rides = self.agent_info.traffic_simulator._rides_per_day_part
        unused_scooters = action * self.agent_info.scooters_num - potential_rides_start * total_rides
        unused_scooters = np.sum(np.maximum(unused_scooters, 0)) / self.agent_info.scooters_num
        return np.array([moving_dist, unused_scooters])



    def get_state(self, end_day_scooters_locations: Map, potential_starts: Map, normalize=True) -> np.ndarray:
        binx: np.ndarray
        biny: np.ndarray
        endpoints = end_day_scooters_locations.get_points()
        startpoints = potential_starts.get_points()

        end_dist = np.sum(110 * (endpoints[None, ...] - self.nest_bins[:, None, :]) **2, axis=-1)
        end_dist_counts = np.bincount(end_dist.argmin(axis=0), minlength=self.n_actions)
        end_dist_counts = end_dist_counts / end_dist_counts.sum()

        start_dist = np.sum(110 * (startpoints[None, ...] - self.nest_bins[:, None, :]) **2, axis=-1)
        start_dist_counts = np.bincount(start_dist.argmin(axis=0), minlength=self.n_actions)
        start_dist_counts = start_dist_counts / start_dist_counts.sum()
        self.avg_start = (1 - self.alpha) * self.avg_start + self.alpha * start_dist_counts
        return np.stack([self.avg_start,
                         end_dist_counts
                         ], axis=-1)
                         # cur_start_means, cur_end_means], axis=-1)
        # return end_day_scooters_locations.get_points(), potential_starts.get_points()
        # state_end = binned_statistic_2d(end_day_scooters_locations.get_points()[..., 0],
        #                             end_day_scooters_locations.get_points()[..., 1],
        #                             None, 'count', bins=[binx, biny]).statistic
        # state_potential_start = binned_statistic_2d(potential_starts[:, 0],
        #                                 potential_starts[:, 1],
        #                                 None, 'count', bins=[binx, biny]).statistic
        # assert end_day_scooters_locations.get_points().shape[0] == int(np.sum(state_end))
        # if normalize:
        #     state_end /= np.sum(state_end)
        #     state_potential_start /= np.sum(state_potential_start)
        #     # print(state_potential_start)
        # state = np.stack([state_end, state_potential_start], axis=-1)
        # return state
        # return np.expand_dims(state_potential_start, axis=-1)


    def get_half_state(self, end_day_scooters_locations: Map, normalize=True) -> np.ndarray:
        binx: np.ndarray
        biny: np.ndarray
        binx, biny = TrafficGenerator.get_coordinates_bins(self.agent_info.grid_len)
        state_end = binned_statistic_2d(end_day_scooters_locations[:, 0],
                                    end_day_scooters_locations[:, 1],
                                    None, 'count', bins=[binx, biny]).statistic
        if normalize:
            state_end /= np.sum(state_end)
        return state_end


    def get_start_state(self, normalize=True) -> (Map, np.ndarray):
        end_day_scooters_locations: Map = TrafficGenerator.get_random_end_day_scooters_locations(self.agent_info.scooters_num)
        return end_day_scooters_locations, np.ones((self.n_actions, 2)) / self.n_actions
        end_day_state = self.get_half_state(end_day_scooters_locations)
        potential_start = np.ones_like(end_day_state)
        potential_start /= np.sum(potential_start)
        state = np.stack([end_day_state, potential_start], axis=-1)
        # return end_day_scooters_locations, np.expand_dims(potential_start, axis=-1)
        # return end_day_scooters_locations, state

    def get_nests_spread(self, action: np.ndarray) -> List[NestAllocation]:
        # we assume that scooters_locations is an action and not a Map type
        # discretize the action-
        discrete_action = self.discretize(action.copy())
        # print(discrete_action)
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(discrete_action)]

    def discretize(self, data: np.ndarray) -> np.ndarray:
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

    def perform_step(self, prev_scooters_locations: Map, action: np.ndarray, options_index) -> (Map, np.ndarray, float, List[Ride]):
        prev_nests_spread: List[NestAllocation] = self.get_nests_spread(action)
        tmp_prev_nests_locations: Map = self.agent_info.traffic_simulator. \
            get_scooters_location_from_nests_spread(prev_nests_spread)

        # get simulation results - rides completed and scooters final location:
        result: Tuple[List[Ride], Map, Map] = self.agent_info. \
            traffic_simulator.get_simulation_result(tmp_prev_nests_locations, options_index)
        rides_completed: List[Ride] = result[0]

        start_points = [ride.orig for ride in rides_completed]
        # plt.scatter([p.x for p in start_points], [p.y for p in start_points])
        # plt.show()
        next_day_locations: Map = result[1]
        potential_starts: Map = result[2]
        next_state: np.ndarray = self.get_state(next_day_locations, potential_starts)

        # compute revenue
        prev_nests_locations: Map = self.agent_info.traffic_simulator. \
            get_scooters_location_from_nests_spread(prev_nests_spread)
        incomes, expenses = self.agent_info.incomes_expenses.calculate_revenue(rides_completed, prev_scooters_locations,
                                                                               prev_nests_locations)
        money = incomes - expenses
        unused_scooters_amount = self.agent_info.scooters_num - len(rides_completed)
        reward = money - unused_scooters_amount * self.unused_scooters_factor
        # print(f'action {action}')
        # print(f'incomes {incomes}, expenses -{expenses}')
        return prev_nests_spread, next_day_locations, next_state, reward, money, rides_completed

    def get_random_nest_spread(self):
        possible_scooters_spread = [comb for comb in
                                    it.combinations_with_replacement(
                                        range(self.agent_info.scooters_num + 1),
                                        len(self.agent_info.optional_nests))
                                    if sum(comb) == self.agent_info.scooters_num]
        return np.array(random.choice(possible_scooters_spread))

    def learn(self, num_games, game_len, visualize, load_checkpoint, pretrain_critic):
        best_score = float('-inf')
        score_history, score_money_history = [], []

        if load_checkpoint:
            evaluate = False
            self.load_models()

        else:
            evaluate = False

        total_critic_loss, total_actor_loss, total_critic_values, total_learn_rewards = [], [], [], []
        options_index = 0
        # #random part
        if pretrain_critic:
            random_critic_loss, random_rewards, random_critic_values, random_actor_loss = [], [], [], []
            random_game_len = 20
            for i in range(200):
                scooters_locations: Map
                state: np.ndarray
                scooters_locations, state = self.get_start_state()
                random_cur_game_critic_loss = []
                for step_idx in range(random_game_len):

                    action = np.random.random((self.n_actions,))
                    action = action / np.sum(action)
                    # action = np.array([0.5, 0.5, 0, 0])
                    pre_nests_spread: List[NestAllocation]
                    next_day_scooters_locations: Map
                    next_state: np.ndarray
                    reward: float
                    rides_completed: List[Ride]
                    pre_nests_spread, next_day_scooters_locations, next_state, reward, money, rides_completed = \
                        self.perform_step(scooters_locations, action, options_index)
                    options_index = (options_index + 1)
                    extra_features = self.extract_extra_features(state, action)
                    if step_idx != 0:
                        self.remember(self.random_memory, state, extra_features, action, reward, next_state)
                    if not evaluate:
                        critic_loss, actor_loss, critic_values, reward_values = \
                        self.learn_batch(self.random_memory, grad_actor=False, grad_critic=True)
                        random_cur_game_critic_loss.append(critic_loss)
                        random_rewards += reward_values
                        random_critic_values += critic_values
                    state = next_state
                    scooters_locations = next_day_scooters_locations
                random_critic_loss += random_cur_game_critic_loss
                print(i, f'critic loss {np.average(np.array(random_cur_game_critic_loss))}')
                if i % 100 == 0:
                    self.save_models()
            self.save_models()
            self.plot_results(total_critic_loss=random_critic_loss, total_critic_values=random_critic_values,
                               total_learn_rewards=random_rewards, total_actor_loss=total_actor_loss)
            return

        #real games
        for i in range(num_games):
            scooters_locations: Map
            state: np.ndarray
            scooters_locations, state = self.get_start_state()

            total_rides: List[List[Ride]] = []
            total_nest_allocations: List[List[NestAllocation]] = []
            total_rewards: List[float] = []

            score, score_money = 0, 0

            for step_idx in range(game_len):
                action: np.ndarray = self.get_action(state, evaluate)
                # action = np.array([1/2, 1/2, 0, 0])
                pre_nests_spread: List[NestAllocation]
                next_day_scooters_locations: Map
                next_state: np.ndarray
                reward: float
                rides_completed: List[Ride]
                pre_nests_spread, next_day_scooters_locations, next_state, reward, money, rides_completed = \
                    self.perform_step(scooters_locations, action, options_index)
                options_index = (options_index + 1)
                total_rides.append(rides_completed)
                total_rewards.append(reward)
                total_nest_allocations.append(pre_nests_spread)
                score += reward
                score_money += money
                extra_features = self.extract_extra_features(state, action)
                if step_idx != 0:
                    self.remember(self.memory, state, extra_features, action, reward, next_state)
                if not evaluate:
                    critic_loss, actor_loss, critic_values, reward_values = \
                        self.learn_batch(self.memory, grad_actor=True, grad_critic=True)
                    total_critic_loss.append(critic_loss)
                    total_actor_loss.append(actor_loss)
                    total_critic_values += critic_values
                    total_learn_rewards += reward_values
                state = next_state
                scooters_locations = next_day_scooters_locations

            # if visualize and (i == num_games - 1):
            #     vis = Visualizer(rides_list=total_rides,
            #                      nests_list=total_nest_allocations,
            #                      revenue_list=total_rewards)
            #     vis.visualise()

            score /= game_len
            score_money /= game_len
            score_history.append(score)
            score_money_history.append(score_money)
            avg_score = np.mean(score_history[-7:])
            avg_score_money = np.mean(score_money_history[-7:])

            # if avg_score > best_score:
            #     best_score = avg_score
            #     if not load_checkpoint:
            #         self.save_models()
            print(action)
            # print(len(rides_completed))
            print('episode ', i, 'score %.5f' % score, 'avg score %.5f' % avg_score, 'money %.5f' % score_money, 'avg money %.5f' % avg_score_money)

        self.plot_results(total_critic_loss=total_critic_loss, total_critic_values=total_critic_values,
                          total_learn_rewards=total_learn_rewards, total_actor_loss=total_actor_loss)
        return

    @staticmethod
    def plot_results(total_critic_loss, total_critic_values, total_learn_rewards, total_actor_loss):
        f, axs = plt.subplots(4)
        axs[0].plot(range(len(total_actor_loss)), total_actor_loss)
        axs[0].title.set_text('actor_loss')
        axs[1].plot(range(len(total_critic_loss)), total_critic_loss)
        axs[1].set_yscale('log')
        axs[1].title.set_text('critic_loss')
        axs[2].plot(range(len(total_critic_values)), total_critic_values)
        axs[2].title.set_text('total_critic_values')
        axs[3].plot(range(len(total_learn_rewards)), total_learn_rewards)
        axs[3].title.set_text('total_rewards')
        plt.show()





