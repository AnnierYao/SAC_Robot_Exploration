import numpy as np
import torch
import os
import pickle
import random
import collections

class Trajectory:
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.her_fb = 0
        self.length = 0
    
    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1
    
    def store_fd(self, her_fb):
        self.her_fb = her_fb

    8

class ReplayMemory:
    # def __init__(self, capacity, seed):
    #     random.seed(seed)
    #     self.capacity = capacity
    #     self.buffer = []
    #     self.position = 0
    def __init__(self, capacity, seed):
        self.buffer = collections.deque(maxlen=capacity)
        random.seed(seed)

    # def push(self, state, action, reward, next_state, done):
    #     if len(self.buffer) < self.capacity:
    #         self.buffer.append(None)
    #     self.buffer[self.position] = (state, action, reward, next_state, done)
    #     self.position = (self.position + 1) % self.capacity

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    # def sample(self, batch_size):
    #     batch = random.sample(self.buffer, batch_size)
    #     # print('batch:', np.shape(batch))
    #     # state, action, reward, next_state, done = map(np.stack, zip(*batch))
    #     state, action, reward, next_state, done = zip(*batch)
    #     return np.array(state).astype(np.float64), action, reward, np.array(next_state).astype(np.float64), done
        
    def sample(self, batch_size, use_her=True,  her_ratio=0.1):
        batch = dict(states=[], actions=[], next_states=[], rewards=[], dones=[])
        for _ in range(batch_size):
            # 取一条轨迹
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state+1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state] 
            done = traj.dones[step_state]

            if use_her:
                if reward > -1:
                    reward += (traj.her_fb/traj.length)*her_ratio
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['next_states'].append(next_state)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states']).astype(np.float64)
        batch['actions'] = np.array(batch['actions'])
        batch['next_states'] = np.array(batch['next_states']).astype(np.float64)

        return batch


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