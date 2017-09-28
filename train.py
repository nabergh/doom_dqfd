from itertools import count
import time
from datetime import date
import pickle
import argparse
import json

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

from utils import ReplayMemory, PreprocessImage, EpsGreedyPolicy, Transition

# Hyperparameters
GAMMA = 0.99
L2_PENALTY = 10e-5
MARGIN = 10.0

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

def load_demos(fname):
    with open(fname, 'rb') as demo_file:
        return pickle.load(demo_file)

def optimize_model(bsz, demo_prop, opt_step):
    demo_samples = int(bsz * demo_prop)
    demo_trans = []
    if demo_samples > 0:
        demo_trans = demos.sample(demo_samples)
    agent_trans = memory.sample(bsz - demo_samples)
    transitions = demo_trans + agent_trans
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
    q_vals = model(state_batch)
    state_action_values = q_vals.gather(1, action_batch)

    next_state_values = Variable(torch.zeros(bsz).cuda())
    next_state_values[non_final_mask] = model(non_final_next_states).data.max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    q_loss = F.mse_loss(state_action_values, expected_state_action_values, size_average=False)

    if demo_prop > 0:
        num_actions = q_vals.size(1)
        margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * MARGIN
        batch_margins = margins[action_batch.data.squeeze().cpu()]
        q_vals = q_vals + Variable(batch_margins).type(dtype)

        supervised_loss = (q_vals.max(1)[0].unsqueeze(1) - state_action_values).pow(2)[:demo_samples].sum()
        loss = q_loss + supervised_loss
    else:
        loss = q_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_value('Average loss', loss.mean().data[0], opt_step)
    log_value('Q loss', q_loss.mean().data[0], opt_step)

    if demo_prop > 0:
        log_value('Supervised loss', supervised_loss.mean().data[0], opt_step)



parser = argparse.ArgumentParser(description='Doom DQN')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--bsz', type=int, default=32, metavar='BSZ',
                    help='batch size (default: 32)')
parser.add_argument('--num-eps', type=int, default=-1, metavar='NE',
                    help='number of episodes to train (default: train forever)')
parser.add_argument('--frame-skip', type=int, default=4, metavar='FS',
                    help='number of frames to skip between agent input (must match frame skip for demos)')
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
parser.add_argument('--eps-start', type=int, default=1.0, metavar='EST',
                    help='starting value for epsilon')
parser.add_argument('--eps-end', type=int, default=0.0, metavar='EEND',
                    help='ending value for epsilon')
parser.add_argument('--eps-steps', type=int, default=6000, metavar='ES',
                    help='number of episodes before epsilon equals eps-end (linearly degrades)')
parser.add_argument('--demo-prop', type=float, default=0.3, metavar='DR',
                    help='proportion of batch to set as transitions from the demo file')
parser.add_argument('--no-train', action="store_true",
                    help='set to true if you don\'t want to actually train')
parser.add_argument('--demo-file', default=None, metavar='DF',
                    help='file to load pickled demonstrations')

api_key = ''
with open('api_key.json', 'r+') as api_file:
    api_key = json.load(api_file)['api_key']


if __name__ == '__main__':
    args = parser.parse_args()
    
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    save_name = args.save_name + '_' + timestring
    run_name = save_name.split('/')[-1]
    configure("logs/" + run_name, flush_secs=5)
    
    env_spec = gym.spec('ppaquette/' + args.env_name)
    env_spec.id = args.env_name
    env = env_spec.make()
    env = ToDiscrete("minimal")(env)
    env = SkipWrapper(4)(env)
    env = PreprocessImage(env)
    if args.monitor:
        env = Monitor(env, 'monitor/' + run_name)
    env.reset()
    memory = ReplayMemory(10000)

    if args.demo_file is not None:
        demos = load_demos(args.demo_file)

    model = DQN(dtype, (1, 80, 80), env.action_space.n)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    if args.no_train:
        args.eps_start = 0.0
        args.eps_end = 0.0

    policy = EpsGreedyPolicy(args.eps_start, args.eps_end, args.eps_steps)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=L2_PENALTY)

    opt_step = 0

    if args.demo_file is not None:
        print('Pre-training')
        for i in range(10000):
            opt_step += 1
            optimize_model(args.bsz, 1.0, opt_step)
        print('Pre-training done')
    else:
        args.demo_prop = 0

    ep_counter = count(1) if args.num_eps < 0 else range(args.num_eps)
    for i_episode in ep_counter:
        state = env.reset()
        total_reward = 0
        for step_n in count():
            q_vals = model(Variable(state.type(dtype), volatile=True)).data
            action = policy.select_action(q_vals, env)
            next_state, reward, done, _ = env.step(action[0])
            reward = torch.Tensor([reward])

            if done:
                next_state = None

            memory.push(Transition(state, action, next_state, reward))
            state = next_state
            
            if len(memory) >= args.init_states and not args.no_train:
                opt_step += 1
                optimize_model(args.bsz, args.demo_prop, opt_step)

            total_reward += reward
            if done:
                print('Finished episode ' + str(i_episode) + ' with reward ' + str(total_reward[0]) + ' after ' + str(step_n) + ' steps')
                log_value('Total Reward', total_reward[0], i_episode)
                break
        if i_episode % 100 == 0 and not args.no_train:
            pickle.dump(model.state_dict(), open(save_name + '.p', 'wb'))
    env.close()
    if args.monitor:
        gym.upload('monitor/' + run_name, api_key=api_key)
