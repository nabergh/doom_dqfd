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
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboard_logger import configure, log_value

from utils import ReplayMemory, PreprocessImage, EpsGreedyPolicy, Transition
from models import DQN

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def load_demos(fname):
    with open(fname, 'rb') as demo_file:
        return pickle.load(demo_file)

def optimize_dqn(bsz, opt_step):
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
    q_vals = model(state_batch)
    state_action_values = q_vals.gather(1, action_batch)

    next_state_values = Variable(torch.zeros(bsz).cuda())
    next_state_values[non_final_mask] = model(non_final_next_states).data.max(1)[0]
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    q_loss = F.mse_loss(state_action_values, expected_state_action_values, size_average=False)

    loss = q_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_value('Average loss', loss.mean().data[0], opt_step)
    log_value('Q loss', q_loss.mean().data[0], opt_step)


def optimize_dqfd(bsz, demo_prop, opt_step):

    # creating the training batch from a fixed proportion of demonstration transitions and agent transitions
    demo_samples = int(bsz * demo_prop)
    demo_trans = []
    if demo_samples > 0:
        demo_trans = demos.sample(demo_samples)
    agent_trans = memory.sample(bsz - demo_samples)
    transitions = demo_trans + agent_trans
    batch = Transition(*zip(*transitions))

    # creating PyTorch tensors for the transitions and calculating the q vals for the actions taken
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action).unsqueeze(1))
    reward_batch = Variable(torch.cat(batch.reward))
    n_reward_batch = Variable(torch.cat(batch.n_reward))
    if USE_CUDA:
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        n_reward_batch = n_reward_batch.cuda()
        non_final_mask = non_final_mask.cuda()
    q_vals = model(state_batch)
    state_action_values = q_vals.gather(1, action_batch)

    # comparing the q values to the values expected using the next states and reward
    next_state_values = Variable(torch.zeros(bsz).cuda())
    next_state_values[non_final_mask] = model(non_final_next_states).data.max(1)[0]
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # calculating the q loss and n-step return loss
    q_loss = F.mse_loss(state_action_values, expected_state_action_values, size_average=False)
    n_step_loss = F.mse_loss(state_action_values, n_reward_batch, size_average=False)

    # calculating the supervised loss
    num_actions = q_vals.size(1)
    margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * args.margin
    batch_margins = margins[action_batch.data.squeeze().cpu()]
    q_vals = q_vals + Variable(batch_margins).type(dtype)
    supervised_loss = (q_vals.max(1)[0].unsqueeze(1) - state_action_values).pow(2)[:demo_samples].sum()


    loss = q_loss + args.lam_sup * supervised_loss + args.lam_nstep * n_step_loss

    # optimization step and logging
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 100)
    optimizer.step()

    log_value('Average loss', loss.mean().data[0], opt_step)
    log_value('Q loss', q_loss.mean().data[0], opt_step)
    log_value('Supervised loss', supervised_loss.mean().data[0], opt_step)
    log_value('N Step Reward loss', n_step_loss.mean().data[0], opt_step)



parser = argparse.ArgumentParser(description='Doom DQfD')

# nn optimization hyperparams
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--bsz', type=int, default=32, metavar='BSZ',
                    help='batch size (default: 32)')

# model saving and loading settings
parser.add_argument('--save-name', default='doom_dqn_model', metavar='FN',
                    help='path/prefix for the filename to save model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='LN',
                    help='path/prefix for the filename to load model\'s parameters')

# RL training hyperparams
parser.add_argument('--env-name', default='DoomBasic-v0', metavar='ENV',
                    help='environment to train on (default: DoomBasic-v0')
parser.add_argument('--num-eps', type=int, default=-1, metavar='NE',
                    help='number of episodes to train (default: train forever)')
parser.add_argument('--frame-skip', type=int, default=4, metavar='FS',
                    help='number of frames to skip between agent input (must match frame skip for demos)')
parser.add_argument('--init-states', type=int, default=1000, metavar='IS',
                    help='number of states to store in memory before training (default: 1000)')
parser.add_argument('--gamma', type=float, default=1000, metavar='GAM',
                    help='reward discount per step (default: 0.99)')

# policy hyperparams
parser.add_argument('--eps-start', type=int, default=1.0, metavar='EST',
                    help='starting value for epsilon')
parser.add_argument('--eps-end', type=int, default=0.0, metavar='EEND',
                    help='ending value for epsilon')
parser.add_argument('--eps-steps', type=int, default=10000, metavar='ES',
                    help='number of episodes before epsilon equals eps-end (linearly degrades)')

# DQfD hyperparams
parser.add_argument('--demo-prop', type=float, default=0.3, metavar='DR',
                    help='proportion of batch to set as transitions from the demo file')
parser.add_argument('--demo-file', default=None, metavar='DF',
                    help='file to load pickled demonstrations')
parser.add_argument('--margin', type=float, metavar='MG',
                    help='margin for supervised loss used in DQfD (must be set)')
parser.add_argument('--lam-sup', type=float, default=1.0, metavar='LS',
                    help='weight of the supervised loss (default 1.0)')
parser.add_argument('--lam-nstep', type=float, default=1.0, metavar='LN',
                    help='weight of the n-step loss (default 1.0)')

# testing/monitoring settings
parser.add_argument('--no-train', action="store_true",
                    help='set to true if you don\'t want to actually train')
parser.add_argument('--monitor', action="store_true",
                    help='whether to monitor results')
parser.add_argument('--upload', action="store_true",
                    help='set this (and --monitor) if you want to upload monitored ' \
                        'results to gym (requires api key in an api_key.json)')




if __name__ == '__main__':
    args = parser.parse_args()
    
    # setting the run name based on the save name set for the model and the timestamp
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    save_name = args.save_name + '_' + timestring
    if args.load_name is None:
        run_name = save_name.split('/')[-1]
    else:
        run_name = args.load_name.split('/')[-1]
    configure("logs/" + run_name, flush_secs=5)
    
    # setting up the environment and the replay buffer(s)
    env_spec = gym.spec('ppaquette/' + args.env_name)
    env_spec.id = args.env_name
    env = env_spec.make()
    env = ToDiscrete("minimal")(env)
    if args.monitor:
        env = Monitor(env, 'monitor/' + run_name, video_callable=lambda ep: ep % 10 == 0)
    env = SkipWrapper(args.frame_skip)(env)
    env = PreprocessImage(env)
    env.reset()
    memory = ReplayMemory(10000)

    if args.demo_file is not None:
        demos = load_demos(args.demo_file)

    # instantiating model and optimizer
    model = DQN(dtype, (1, 80, 80), env.action_space.n)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name, 'rb')))
    if not args.no_train:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # instantiating policy object
    if args.no_train:
        args.eps_start = 0.0
        args.eps_end = 0.0
        args.eps_steps = 1

    policy = EpsGreedyPolicy(args.eps_start, args.eps_end, args.eps_steps)

    opt_step = 0

    # pre-training
    if args.demo_file is not None and not args.no_train:
        print('Pre-training')
        for i in range(1000):
            opt_step += 1
            optimize_dqfd(args.bsz, 1.0, opt_step)
        print('Pre-training done')
    else:
        args.demo_prop = 0

    # training loop
    ep_counter = count(1) if args.num_eps < 0 else range(args.num_eps)
    for i_episode in ep_counter:
        state = env.reset()
        total_reward = 0
        transitions = []
        q_vals = model(Variable(state.type(dtype), volatile=True)).data
        for step_n in count():

            # selecting an action and playing it
            if args.no_train:
                action = q_vals.max(1)[1].cpu()
            else:
                action = policy.select_action(q_vals, env)
            next_state, reward, done, _ = env.step(action[0])
            reward = torch.Tensor([reward])

            # storing the transition in a temporary replay buffer which is held in order to calculate n-step returns
            transitions.insert(0, Transition(state, action, next_state, reward, torch.zeros(1)))
            state = next_state        
            gamma = 1
            new_trans = []
            for trans in transitions:
                new_trans.append(trans._replace(n_reward= trans.n_reward + gamma * reward))
                gamma = gamma * args.gamma
            transitions = new_trans

            # if the episode isn't over, get the next q vals and add the 10th transition to the replay buffer
            # (this algorithm uses 10-step returns)
            # otherwise push all transitions to the buffer
            if not done:
                q_vals = model(Variable(next_state.type(dtype), volatile=True)).data

                if len(transitions) >= 10:
                    last_trans = transitions.pop()
                    last_trans = last_trans._replace(n_reward=last_trans.n_reward + gamma * q_vals.max(1)[0].cpu())
                    memory.push(last_trans)

                state = next_state
            
            else:
                for trans in transitions:
                    memory.push(trans)

            # optimization step for the network the network
            if len(memory) >= args.init_states and not args.no_train:
                opt_step += 1
                optimize_dqfd(args.bsz, args.demo_prop, opt_step)

            # logging
            total_reward += reward
            if done:
                print('Finished episode ' + str(i_episode) + ' with reward ' + str(total_reward[0]) + ' after ' + str(step_n) + ' steps')
                log_value('Total Reward', total_reward[0], i_episode)
                break
        # saving the model every 100 episodes
        if i_episode % 100 == 0 and not args.no_train:
            pickle.dump(model.state_dict(), open(save_name + '.p', 'wb'))
    env.close()

    # uploading results to gym (api_key.json required) although at the time of this code being written, the Gym website was read-only =\
    if args.upload and args.monitor:
        api_key = ''
        with open('api_key.json', 'r+') as api_file:
            api_key = json.load(api_file)['api_key']
        gym.upload('monitor/' + run_name, api_key=api_key)
