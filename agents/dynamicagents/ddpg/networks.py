import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, MaxPool2D
from agents.agent import AgentInfo
from scipy.stats import binned_statistic_2d
from data.trafficgenerator import TrafficGenerator
from data.trafficdatatypes import *

class CriticNetwork(keras.Model):
    def __init__(self, name='critic', chkpt_dir=r"""C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters"""):
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

    def call(self, state, action):
        # state_res = self.state_conv1(state)
        # state_res = self.state_conv2(state_res)
        # state_res = self.state_conv3(state_res)
        state_res_flat = self.state_flat(state)
        # action_res = self.action_fc1(action)
        # action_res = self.action_fc2(action_res)
        state_action_res = self.state_action_concat([state_res_flat, action])

        # state_action_res = self.state_action_concat([state_res_flat, action_res])
        state_action_res = self.both_fc1(state_action_res)
        # state_action_res = self.both_fc2(state_action_res)
        value = self.final(state_action_res)
        return value


class ActorNetwork(keras.Model):
    def __init__(self,  n_actions, name='actor',
                 chkpt_dir=r"""C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters"""):
        super(ActorNetwork, self).__init__()
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
        self.d1 = Dense(4 * n_actions, activation='relu', kernel_regularizer='l1')
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