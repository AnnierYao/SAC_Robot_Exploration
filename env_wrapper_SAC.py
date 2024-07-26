import gymnasium as gym
from utilis.default_config import default_config
from gymnasium import spaces
import numpy as np
from skimage.transform import resize
import robot_simulation as robot
import matplotlib.pyplot as plt
import csv
import os

SNAKE_LEN_GOAL = 30
# training environment parameters
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e4  # timesteps to observe before training
EXPLORE = 5e5  # frames over which to anneal epsilon
REPLAY_MEMORY = 10000  # number of previous transitions to remember
BATCH = 64  # size of minibatch
FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate
TARGET_UPDATE = 1e2  # update frequency of the target network

# select mode
TRAIN = True
PLOT = True

class RobEnv(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self):
		super(RobEnv, self).__init__()
		self.total_reward = np.empty([0, 0])
		self.map_step = 0
		self.map_index = 1
		self.test_index = 1
		self.totoal_step = 0
		self.num_timesteps = 0
		self.done = False
		self._max_episode_steps = 100
		self.infos = {'terminal': [], 're_locate': [], 'collision_index': [], 'finish_all_map': [], 
			'new_average_reward': [], 'total_step': [], 'map_step': [], 'map_index': [], 'total_reward': [], 'done':{}, 'num_timesteps':[]}
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		# self.action_space = spaces.Discrete(ACTIONS)
		self.action_space = spaces.Box(low=-40, high=40, shape=(2,), dtype=np.float64)
		# Example for using image as input (channel-first; channel-last also works):
		self.robot_explo = robot.Robot(0, TRAIN, PLOT)
		self.observation_space = spaces.Box(low=0, high=1, shape=(57600,), dtype=np.uint8)


	def step(self, action, test=False):
                test_path = './img/A2c/test_{}_{}_{}'.format(default_config.replay_size, default_config.lr, default_config.batch_size)
                train_path = './img/A2c/train_{}_{}_{}'.format(default_config.replay_size, default_config.lr, default_config.batch_size)
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                state, self.reward, terminal, self.done, re_locate, collision_index, finish_all_map = self.robot_explo.step(action)
		# self.observation = state.flatten()
                self.observation = (state/255).flatten()
                # state = resize(state/255, (84, 84))
		# self.observation = np.reshape(state, (84, 84, 1))
                self.map_step += 1
                self.num_timesteps += 1
                self.total_reward = np.append(self.total_reward, self.reward)
                truncated = finish_all_map
                if re_locate:
                    state, self.done, finish_all_map = self.robot_explo.rescuer()
                    self.observation = (state/255).flatten()
                if finish_all_map:
                    self.done = True
                if self.done == True:
                    self.totoal_step += self.map_step
                    if test:
                        plt.savefig(test_path + "/%s-%d.png" % (self.test_index, self.map_step), format='png')
                        self.map_step = 0
                        self.test_index += 1
                    else:
                            plt.savefig(train_path + "/%s-%d.png" % (self.map_index, self.map_step), format='png')
                            self.map_step = 0
                            self.map_index += 1
                if self.num_timesteps > OBSERVE:
                    new_average_reward = np.average(self.total_reward[len(self.total_reward) - 100:])
                    self.infos['new_average_reward'].append(new_average_reward)
                
                self.infos['terminal']=terminal
                self.infos['re_locate']=re_locate
                self.infos['collision_index']=collision_index
                self.infos['finish_all_map']=finish_all_map
                self.infos['total_step']=self.totoal_step
                self.infos['map_step']=self.map_step
                self.infos['map_index']=self.map_index
                self.infos['total_reward']=self.total_reward
                self.infos['done']=self.done
                self.infos['num_timesteps']=self.num_timesteps
            
                return self.observation, self.reward, self.done, truncated, self.infos
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.done = False
		state = self.robot_explo.begin()
		self.observation = (state/255).flatten()
		return self.observation, self.infos  # reward, done, info can't be included
	
	def render(self, mode="human", close=False):

		...
