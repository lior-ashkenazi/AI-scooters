import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from scipy.stats import binned_statistic_2d
from typing import List, Tuple

from agents.dynamicagents.ddpg.buffer import ReplayBuffer
from agents.dynamicagents.ddpg.networks import ActorNetwork, CriticNetwork
# from agents.dynamicagents.ddpg.utils import plot_learning_curve


from agents.dynamicagent import DynamicAgent
from agents.rlagent import ReinforcementLearningAgent
from data.trafficdatatypes import Map, NestAllocation, Ride
from data.trafficgenerator import TrafficGenerator


class DdpgAgent(ReinforcementLearningAgent, DynamicAgent):
    def __init__(self, env_agent_info, actor_lr=1e-4, critic_lr=2e-5,
                 decay_factor=0.99, max_size=2000, target_update_rate=1e-3,
                 batch_size=64, noise=0.1):
        super(DdpgAgent, self).__init__(env_agent_info)
        self.decay_factor = decay_factor
        self.target_update_rate = target_update_rate
        input_shape = (self.agent_info.grid_len, self.agent_info.grid_len)
        self.n_actions = len(self.agent_info.optional_nests)
        self.memory = ReplayBuffer(max_size, input_shape, self.n_actions)
        self.batch_size = batch_size
        self.n_actions = self.n_actions
        self.min_action = 0
        self.max_action = 1
        self.noise = noise

        self.actor = ActorNetwork(n_actions=self.n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=self.n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.update_network_parameters(target_update_rate=1)

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

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state, action, reward, next_state)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def get_action(self, state, evaluate=False):
        actor_input = tf.convert_to_tensor([state], dtype=tf.float32)
        actor_input = tf.expand_dims(actor_input, axis=-1)
        action = self.actor(actor_input)
        action = action.numpy()
        if not evaluate:
            action += np.random.normal(scale=self.noise, size=self.n_actions)
            # action += tf.random.normal(shape=[self.n_actions],
            #                            mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        action = tf.clip_by_value(action, self.min_action, self.max_action)[0]
        if np.all(action == 0):
            action = np.full(action.shape, 1)
        action /= np.sum(action)
        return action

    def learn_batch(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, next_state = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states = tf.expand_dims(states, axis=-1)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        next_states = tf.expand_dims(next_states, axis=-1)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            critic_value_next = tf.squeeze(self.target_critic(
                                next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.decay_factor * critic_value_next
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def get_state(self, end_day_scooters_locations: Map, normalize=True) -> np.ndarray:
        binx: np.ndarray
        biny: np.ndarray
        binx, biny = TrafficGenerator.get_coordinates_bins(self.agent_info.grid_len)
        state = binned_statistic_2d(end_day_scooters_locations[:, 0],
                                    end_day_scooters_locations[:, 1],
                                    None, 'count', bins=[binx, biny]).statistic
        assert end_day_scooters_locations.get_points().shape[0] == int(np.sum(state))
        if normalize:
            state /= np.sum(state)
        return state

    def get_start_state(self, normalize=True) -> (Map, np.ndarray):
        start_scooters_locations: Map = TrafficGenerator.get_random_end_day_scooters_locations(
            self.agent_info.scooters_num)
        start_state: np.ndarray = self.get_state(start_scooters_locations, normalize=normalize)
        return start_scooters_locations, start_state

    def get_nests_spread(self, action: np.ndarray) -> List[NestAllocation]:
        # we assume that scooters_locations is an action and not a Map type
        # discretize the action-
        discrete_action = self.discretize(action)
        return [NestAllocation(self.agent_info.optional_nests[i], scooters_num)
                for i, scooters_num in enumerate(discrete_action)]

    def discretize(self, data: np.ndarray) -> np.ndarray:
        assert np.isclose(np.sum(data), 1)
        data *= self.agent_info.scooters_num
        data_fraction, data_int = np.modf(data)

        round_up_amount = np.sum(data_fraction)
        assert np.isclose(round_up_amount, round(round_up_amount))
        round_up_amount = round(round_up_amount)

        data_fraction_flat, data_int_flat = data_fraction.flatten(), data_int.flatten()
        data_fraction_index = np.argsort(-data_fraction_flat)
        data_int_flat[data_fraction_index[:round_up_amount]] += 1
        data_discrete = data_int_flat.reshape(data.shape)
        assert np.sum(data_discrete) == self.agent_info.scooters_num
        data_discrete = data_discrete.astype(int)
        return data_discrete

    def perform_step(self, prev_scooters_locations: Map, action: np.ndarray) -> (Map, np.ndarray, float, List[Ride]):
        prev_nests_spread: List[NestAllocation] = self.get_nests_spread(action)
        prev_nests_locations: Map = self.agent_info.traffic_simulator. \
            get_scooters_location_from_nests_spread(prev_nests_spread)

        # get simulation results - rides completed and scooters final location:
        result: Tuple[List[Ride], Map] = self.agent_info. \
            traffic_simulator.get_simulation_result(prev_nests_locations)
        rides_completed: List[Ride] = result[0]
        next_day_locations: Map = result[1]
        next_state: np.ndarray = self.get_state(next_day_locations)

        # compute revenue
        reward: float = self.agent_info.incomes_expenses.calculate_revenue(
            rides_completed, prev_scooters_locations, prev_nests_locations)
        return next_day_locations, next_state, reward, rides_completed

    def learn(self):
        n_games = 100
        game_len = 50
        best_score = float('-inf')
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            evaluate = True
            self.load_models()

        else:
            evaluate = False

        for i in range(n_games):
            scooters_locations: Map
            state: np.ndarray
            scooters_locations, state = self.get_start_state()
            score = 0

            for step_idx in range(game_len):
                action: np.ndarray = self.get_action(state, evaluate)
                next_day_scooters_locations: Map
                next_state: np.ndarray
                reward: float
                rides_completed: List[Ride]
                next_day_scooters_locations, next_state, reward, rides_completed = self.perform_step(scooters_locations, action)
                score += reward
                self.remember(state, action, reward, next_state)
                if not evaluate:
                    self.learn_batch()
                state = next_state
                scooters_locations = next_day_scooters_locations
            score /= n_games
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    self.save_models()
            print('episode ', i, 'score %.5f' % score, 'avg score %.5f' % avg_score)
        return





