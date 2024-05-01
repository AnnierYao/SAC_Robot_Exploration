import numpy as np
import torch
import os
import pickle
import random

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # print('batch:', np.shape(batch))
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        state, action, reward, next_state, done = zip(*batch)

        # print('state:', np.shape(state[0]))
        # print('action:', np.shape(action))
        # print('reward:', np.shape(reward))
        # print('next_state:', np.shape(next_state[0]))
        # print('done:', np.shape(done))

        # state_flat = np.array(state).flatten()
        # next_state_flat = np.array(next_state).flatten()

        # print('state:', np.shape(state), np.shape(state_flat), state_flat.dtype)
        # print('next_state:', np.shape(next_state), np.shape(next_state_flat), next_state_flat.dtype)

        # print(state)

        # 将数组转换为指定的数据类型
        # state_batch = state_flat.astype(np.float64)
        # next_state_batch = next_state_flat.astype(np.float64)
        # print('state:', np.shape(state_batch))
        # print('next_state:', np.shape(next_state_batch))
        return np.array(state).astype(np.float64), action, reward, np.array(next_state).astype(np.float64), done
    
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path, i_episode):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity