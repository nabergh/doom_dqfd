from collections import namedtuple
import numpy as np
from itertools import count
import random
import math
import time
from datetime import date

import gym
from gym.core import ObservationWrapper
import ppaquette_gym_doom
from doom_utils import ToDiscrete

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image

from tensorboard_logger import configure, log_value

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 200

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class PreprocessImage(ObservationWrapper):
	def __init__(self, env, height=96, width=128):
		"""A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
		super(PreprocessImage, self).__init__(env)
		self.img_size = (height, width)
		self.resize = T.Compose([T.ToPILImage(),
					T.Scale(96, interpolation=Image.CUBIC),
					T.ToTensor()])

# image transformations, to a 1x3x128x96 tensor
	def _observation(self, screen):
		screen = torch.from_numpy(screen)
		screen = screen.transpose(0, 2)
		screen = self.resize(screen)
		screen = screen.unsqueeze(0)
		return screen.type(dtype)

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None) 
			self.memory[self.position] = Transition(*args)
			self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

steps_done = 0


class DQN(nn.Module):
	def __init__(self, dtype, num_actions):
		super(DQN, self).__init__()
		self.dtype = dtype
		self.pool = nn.MaxPool2d(2, stride=2)

		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

		self.lin1 = nn.Linear(12288, 1024)
		self.lin2 = nn.Linear(1024, 256)
		self.lin3 = nn.Linear(256, num_actions)

		self.type(dtype)
		self.steps_done = 0

	def forward(self, states):
		x = self.pool(F.leaky_relu(self.conv1(states)))
		x = self.pool(F.leaky_relu(self.conv2(x)))
		x = self.pool(F.leaky_relu(self.conv3(x)))
		x = self.pool(F.leaky_relu(self.conv4(x)))

		# flattening each element in the batch
		x = x.view(states.size(0), -1)

		x = F.leaky_relu(self.lin1(x))
		x = F.leaky_relu(self.lin2(x))
		return self.lin3(x)

	def select_action(self, state, env):
		sample = random.random()
		eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
		self.steps_done += 1
		if sample > eps_threshold:
			return self(Variable(state.type(self.dtype), volatile=True)).data.max(1)[1].cpu()
		else:
			return torch.LongTensor([[env.action_space.sample()]])

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).cuda()
	non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
	non_final_next_states = Variable(non_final_next_states_t, volatile=True)
	state_batch = Variable(torch.cat(batch.state))
	action_batch = Variable(torch.cat(batch.action))
	reward_batch = Variable(torch.cat(batch.reward))
	if USE_CUDA:
		state_batch = state_batch.cuda()
		action_batch = action_batch.cuda()
		reward_batch = reward_batch.cuda()
	state_action_values = model(state_batch).gather(1, action_batch)

	next_state_values = Variable(torch.zeros(BATCH_SIZE).cuda())
	next_state_values[non_final_mask] = model(non_final_next_states).data.max(1)[0].squeeze()[non_final_mask]
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
	log_value('Average loss', loss.mean().data[0], model.steps_done)
	
	optimizer.zero_grad()
	loss.backward()

timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
run_name = "DoomBasic_simpledqn" + '_' + timestring
configure("logs/" + run_name, flush_secs=5)

DOOM_ENV = 'DoomBasic-v0'
env = gym.make('ppaquette/' + DOOM_ENV)
env = ToDiscrete("minimal")(env)
env = PreprocessImage(env)
env.reset()
memory = ReplayMemory(1000)
model = DQN(dtype, env.action_space.n)
optimizer = optim.Adam(model.parameters())

for i_episode in count(1):

	state = env.reset()
	total_reward = 0

	for t in count():
		# action = env.action_space.sample()
		action = model.select_action(state, env)
		next_state, reward, done, _ = env.step(action[0,0])
		reward = torch.Tensor([reward])

		memory.push(state, action, next_state, reward)

		state = next_state

		optimize_model()

		total_reward += reward
		if done:
			print(f'Finished episode {i_episode} with reward {total_reward} after {t} steps')
			log_value('Total Reward', total_reward[0], i_episode)
			break

# env.close()
