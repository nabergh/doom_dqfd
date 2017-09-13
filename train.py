import numpy as np
from itertools import count
import random
import math
import time
from datetime import date
import pickle
import argparse

import gym
import ppaquette_gym_doom
from doom_utils import ToDiscrete
from gym.wrappers import SkipWrapper, Monitor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboard_logger import configure, log_value

from utils import ReplayMemory, PreprocessImage

# Hyperparameters
GAMMA = 0.99
EPS_START = 1.00
EPS_END = 0.00
EPS_STEPS = 6000

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

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
        if sample > self.epsilons[min(self.steps_done, EPS_STEPS - 1)] or args.no_train:
            return self(Variable(state.type(self.dtype), volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.LongTensor([env.action_space.sample()])

def optimize_model(bsz):
    if len(memory) < bsz:
        return
    transitions = memory.sample(bsz)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action).unsqueeze(1))
    reward_batch = Variable(torch.cat(batch.reward))
    if USE_CUDA:
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        non_final_mask = non_final_mask.cuda()
    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(bsz).cuda())
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
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--bsz', type=int, default=32, metavar='BSZ',
                    help='batch size (default: 32)')
parser.add_argument('--num-eps', type=int, default=-1, metavar='NE',
                    help='number of episodes to train (default: train forever)')
parser.add_argument('--init-states', type=int, default=1000, metavar='IS',
                    help='number of states to store in memory before training (default: 1000)')
parser.add_argument('--env-name', default='DoomBasic-v0', metavar='ENV',
                    help='environment to train on (default: DoomBasic-v0')
parser.add_argument('--save-name', default='doom_dqn_model', metavar='FN',
                    help='path/prefix for the filename to save model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='SN',
                    help='path/prefix for the filename to load model\'s parameters')
parser.add_argument('--monitor', action="store_true",
                    help='whether to monitor results')
parser.add_argument('--no-train', action="store_true",
                    help='set to true if you don\'t want to actually train')

if __name__ == '__main__':
    args = parser.parse_args()
    
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    save_name = args.save_name + '_' + timestring
    run_name = save_name.split('/')[-1]
    configure("logs/" + run_name, flush_secs=5)

    env = gym.make('ppaquette/' + args.env_name)
    env = ToDiscrete("minimal")(env)
    env = SkipWrapper(4)(env)
    env = PreprocessImage(env)
    if args.monitor:
        env = Monitor(env, 'monitor/' + run_name)
    env.reset()
    memory = ReplayMemory(10000)

    model = DQN(dtype, (1, 80, 80), env.action_space.n)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.epsilons = np.linspace(EPS_START, EPS_END, EPS_STEPS)

    ep_counter = count(1) if args.num_eps < 0 else range(args.num_eps)
    for i_episode in ep_counter:
        state = env.reset()
        total_reward = 0
        for step_n in count():
            action = model.select_action(state, env)
            next_state, reward, done, _ = env.step(action[0])
            reward = torch.Tensor([reward])

            if done:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            if len(memory) >= args.init_states and not args.no_train:
                optimize_model(bsz)

            total_reward += reward
            if done:
                print('Finished episode ' + str(i_episode) + ' with reward ' + str(total_reward[0]) + ' after ' + str(step_n) + ' steps')
                log_value('Total Reward', total_reward[0], i_episode)
                break
        if i_episode % 100 == 0 and not args.no_train:
            pickle.dump(model.state_dict(), open(save_name + '.p', 'wb'))