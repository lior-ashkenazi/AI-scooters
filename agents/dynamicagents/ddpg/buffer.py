import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.extra_features_memory = np.zeros((self.mem_size, 2))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, extra_features, action, reward, state_):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.extra_features_memory[index] = extra_features
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        extra_features = self.extra_features_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, extra_features, actions, rewards, next_states

