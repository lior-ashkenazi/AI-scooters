import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate


class CriticNetwork(keras.Model):
    def __init__(self, name='critic', chkpt_dir=r'C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters\agents\dynamicagents\ddpg\ckpts\tmp'):
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        self.action_fc1 = Dense(30, activation='relu')
        self.action_fc2 = Dense(20, activation='relu')

        self.state_conv1 = Conv2D(filters=4, kernel_size=3, strides=1, activation='relu')
        self.state_conv2 = Conv2D(filters=8, kernel_size=3, strides=2, activation='relu')
        self.state_conv3 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')
        self.state_redu = Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')
        self.state_flat = Flatten()

        self.state_action_concat = Concatenate()
        self.both_fc1 = Dense(20, activation='relu')
        self.both_fc2 = Dense(15, activation='relu')
        self.final = Dense(1, activation=None)

    def call(self, state, action):
        state_res = self.state_conv1(state)
        state_res = self.state_conv2(state_res)
        state_res = self.state_conv3(state_res)
        state_res = self.state_redu(state_res)
        state_res_flat = self.state_flat(state_res)

        action_res = self.action_fc1(action)
        action_res = self.action_fc2(action_res)

        state_action_res = self.state_action_concat([state_res_flat, action_res])
        state_action_res = self.both_fc1(state_action_res)
        state_action_res = self.both_fc2(state_action_res)
        value = self.final(state_action_res)
        return value


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name='actor',
            chkpt_dir=r'C:\Users\yonathanb\Desktop\studies\year3\semester2\ai\exercises\practical\AI-scooters\agents\dynamicagents\ddpg\ckpts\tmp'):
        super(ActorNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.h5')

        self.conv1 = Conv2D(filters=4, kernel_size=3, strides=1, activation='relu')
        self.conv2 = Conv2D(filters=8, kernel_size=3, strides=2, activation='relu')
        self.conv3 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')
        self.redu = Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')
        self.flat = Flatten()
        self.final = Dense(n_actions, activation='softmax')

    def call(self, state):
        res = self.conv1(state)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.redu(res)
        res_flat = self.flat(res)
        action = self.final(res_flat)
        return action