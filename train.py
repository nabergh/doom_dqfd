from collections import namedtuple
import numpy as np
from itertools import count
import random
import math
import time
from datetime import date
import pickle
import argparse

import gym
from gym.core import ObservationWrapper
import ppaquette_gym_doom
from doom_utils import ToDiscrete
from gym.wrappers import SkipWrapper

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
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.00
EPS_END = 0.00
EPS_STEPS = 6000

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, width=80, height=80):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.resize = T.Compose([T.ToPILImage(),
                    T.Scale((width,height), interpolation=Image.CUBIC),
                    T.ToTensor()])
        self.normalize = T.Normalize((0, 0, 0), (1, 1, 1))

# image transformations, to a 1x3x80x80 tensor
    def _observation(self, screen):
        screen = torch.from_numpy(screen)
        screen = screen.permute(2, 1, 0)
        screen = self.resize(screen)
        screen = screen.mean(0, keepdim=True)
        # screen = self.normalize(screen)
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

class DQN(nn.Module):
    def __init__(self, dtype, input_shape, num_actions):
        super(DQN, self).__init__()
        self.dtype = dtype
        # self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        conv_out_size = self._get_conv_output(input_shape)

        self.lin1 = nn.Linear(conv_out_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_actions)

        self.type(dtype)
        self.steps_done = -1

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        print("Conv out shape: %s" % str(output_feat.size()))
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x

    def forward(self, states):
        x = self._forward_conv(states)

        # flattening each element in the batch
        x = x.view(states.size(0), -1)

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        return self.lin3(x)

    def select_action(self, state, env):
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        sample = random.random()
        self.steps_done += 1
        if sample > self.epsilons[min(self.steps_done, EPS_STEPS - 1)]:
            return self(Variable(state.type(self.dtype), volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.LongTensor([env.action_space.sample()])

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).cuda()
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action).unsqueeze(1))
    reward_batch = Variable(torch.cat(batch.reward))
    if USE_CUDA:
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).cuda())
    next_state_values[non_final_mask] = model(non_final_next_states).data.max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    log_value('Average loss', loss.mean().data[0], model.steps_done)
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()




# env.close()

parser = argparse.ArgumentParser(description='Doom DQN')
# parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
#                     help='learning rate (default: 0.0001)')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor for rewards (default: 0.99)')
# parser.add_argument('--tau', type=float, default=1.00, metavar='T',
#                     help='parameter for GAE (default: 1.00)')
# parser.add_argument('--beta', type=float, default=0.01, metavar='B',
#                     help='parameter for entropy (default: 0.01)')
# parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
#                     help='number of forward steps in A3C (default: 20)')
# parser.add_argument('--env-name', default='Breakout-ram-v0', metavar='ENV',
#                     help='environment to train on (default: Breakout-ram-v0)')
parser.add_argument('--save-name', default='a3c_model', metavar='FN',
                    help='path/prefix for the filename to save model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='SN',
                    help='path/prefix for the filename to load model\'s parameters')
# parser.add_argument('--evaluate', action="store_true",
#                     help='whether to evaluate results and upload to gym')

if __name__ == '__main__':
    args = parser.parse_args()
    
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    save_name = args.save_name + '_' + timestring
    configure("logs/" + save_name.split('/')[-1], flush_secs=5)

    DOOM_ENV = 'DoomBasic-v0'
    env = gym.make('ppaquette/' + DOOM_ENV)
    env = ToDiscrete("minimal")(env)
    env = SkipWrapper(4)(env)
    env = PreprocessImage(env)
    env.reset()
    memory = ReplayMemory(10000)
    model = DQN(dtype, (1, 80, 80), env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.epsilons = np.linspace(EPS_START, EPS_END, EPS_STEPS)

    for i_episode in count(1):
        state = env.reset()
        total_reward = 0
        for t in count():
            action = model.select_action(state, env)
            next_state, reward, done, _ = env.step(action[0])
            reward = torch.Tensor([reward])

            if done:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            if len(memory) >= 1000:
                optimize_model()

            total_reward += reward
            if done:
                print('Finished episode ' + str(i_episode) + ' with reward ' + str(total_reward[0]) + ' after ' + str(t) + ' steps')
                log_value('Total Reward', total_reward[0], i_episode)
                break
        if i_episode % 100 == 0:
            pickle.dump(model.state_dict(), open(save_name + '.p', 'wb'))