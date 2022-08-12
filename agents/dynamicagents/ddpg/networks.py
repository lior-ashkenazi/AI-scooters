import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, MaxPool2D
from agents.agent import AgentInfo
from scipy.stats import binned_statistic_2d
from data.trafficgenerator import TrafficGenerator
from data.trafficdatatypes import *

class CriticNetwork(keras.Model):
    def __init__(self, name='critic', chkpt_dir=r"""C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters\models\simple_3"""):
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        # self.action_fc1 = Dense(30, activation='relu')
        # self.action_fc2 = Dense(20, activation='relu')

        # self.state_conv1 = ConvLayer(filters=2)
        # self.state_conv2 = ConvLayer(filters=4)
        # self.state_conv3 = ConvLayer(filters=8)
        self.state_flat = Flatten()

        self.state_action_concat = Concatenate()
        self.both_fc1 = Dense(5, activation='relu', kernel_regularizer='l1')
        self.final = Dense(1, activation=None)

    def call(self, state, action, optimal_transport_approx, unused_scooters_percent):
        # state_res = self.state_conv1(state)
        # state_res = self.state_conv2(state_res)
        # state_res = self.state_conv3(state_res)
        state_res_flat = self.state_flat(state)
        # action_res = self.action_fc1(action)
        # action_res = self.action_fc2(action_res)
        state_action_res = self.state_action_concat([state_res_flat, action, optimal_transport_approx, unused_scooters_percent])

        # state_action_res = self.state_action_concat([state_res_flat, action_res])
        state_action_res = self.both_fc1(state_action_res)
        # state_action_res = self.both_fc2(state_action_res)
        value = self.final(state_action_res)
        return value


class ActorNetwork(keras.Model):
    def __init__(self,  n_actions, scooters_num, rides_num, name='actor',
                 chkpt_dir=r"""C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters\models\simple_3"""):
        super(ActorNetwork, self).__init__()
        self.scooters_num = scooters_num
        self.rides_num = rides_num
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        # self.nest_bins = get_nest_bins(env_agent_info)

        # self.conv1 = ConvLayer(filters=2)
        # self.conv2 = ConvLayer(filters=4)
        # self.conv3 = ConvLayer(filters=8)
        # self.flat = Flatten()
        # self.before_final = Dense(2 * n_actions, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(4 * n_actions, activation='relu')
        self.final = Dense(n_actions, activation='softmax')

    def call(self, state):
        # to_input = state_to_input(state)

        # res = self.conv1(state)
        # res = self.conv2(res)
        # res = self.conv3(res)
        # res_flat = self.flat(res)
        # res_flat = self.before_final(res_flat)
        res = self.flatten(state)
        res = self.d1(res)
        action = self.final(res)
        return action

    def _create_nest_bins(self, env_agent_info: AgentInfo):
        nests = env_agent_info.optional_nests
        binx, biny = TrafficGenerator.get_coordinates_bins(env_agent_info.grid_len)
        nest_bins = np.stack([np.digitize([n.x for n in nests], bins=binx),
                              np.digitize([n.y for n in nests], bins=biny)], axis=-1) - 1
        return nest_bins

    def _state_to_input(self, state):

        return state

    def extract_extra_features(self, state, action):
        state = tf.cast(state, tf.float32)
        action = tf.cast(action, tf.float32)
        potential_start = tf.gather(state, indices=0, axis=2)
        potential_start = tf.math.scalar_mul(self.rides_num, potential_start)
        action_scooters = tf.math.scalar_mul(self.scooters_num, action)
        unused_scooters = tf.math.subtract(action_scooters, potential_start)
        unused_scooters = tf.math.maximum(unused_scooters, 0)
        unused_scooters_sum = tf.math.reduce_sum(unused_scooters, axis=1)
        unused_scooters_percent = tf.math.divide(unused_scooters_sum, self.scooters_num)
        unused_scooters_percent = tf.expand_dims(unused_scooters_percent, axis=-1)

        end_day_locations = tf.gather(state, indices=1, axis=2)
        end_nest_dist = tf.math.abs(tf.math.subtract(end_day_locations, action))
        optimal_transport_approx = tf.math.reduce_sum(end_nest_dist, axis=1)
        optimal_transport_approx = tf.expand_dims(optimal_transport_approx, axis=-1)
        return optimal_transport_approx, unused_scooters_percent




class ConvLayer():
    def __init__(self, filters, dropout=0.3):
        self.first = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')
        # self.second = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')
        # self.bn = tf.keras.layers.BatchNormalization()
        # self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0 else None
        self.max_pool = MaxPool2D()

    def __call__(self, inputs):
        res = self.first(inputs)
        # res = self.second(res)
        # res = self.bn(res)
        res = self.max_pool(res)
        return res