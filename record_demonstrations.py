import sys, gym
import ppaquette_gym_doom
from doom_utils import ToDiscrete
from utils import ReplayMemory, PreprocessImage, Transition
from datetime import date
import time
import pickle
import torch
import argparse

GAMMA = 0.99
FRAMESKIP = 4

class TransitionSaver:
    def __init__(self):
        self.processor = PreprocessImage(None)
        self.memory = ReplayMemory()
        self.transitions = []

    def new_episode(self, first_state):
        self.state = self.processor._observation(first_state)

    def add_transition(self, action, next_state, reward):
        if next_state is not None:
            next_state = self.processor._observation(next_state)
            self.transitions.insert(0, Transition(self.state, self.add_noop(action), next_state, torch.FloatTensor([reward]), torch.zeros(1)))

            transitions = []
            gamma = 1
            for trans in self.transitions:
                transitions.append(trans._replace(n_reward= trans.n_reward + gamma * reward))
                gamma = gamma * GAMMA
            self.transitions = transitions
        else:
            for trans in self.transitions:
                self.memory.push(trans)
            self.transitions = []
        self.state = next_state
    
    def add_noop(self, actions):
        actions.insert(0, 0)
        actions = torch.LongTensor(actions)
        actions[0] = (1 - actions[1:].max(0)[0])[0]
        return actions.max(0)[1]

    def save(self, fname):
        with open(fname, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)


def _play_human_mode(self):
    state = self.game.get_state().image_buffer.copy()
    saver.new_episode(state)
    while not self.game.is_episode_finished():
        self.game.advance_action(FRAMESKIP)
        img = self.game.get_state().image_buffer
        if img is not None:
            state = img.copy()
        if self.game.is_episode_finished():
            state = None
        action = self.game.get_last_action()
        reward = self.game.get_last_reward()
        saver.add_transition(action, state, reward)
        time.sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
    return
ppaquette_gym_doom.doom_env.DoomEnv._play_human_mode = _play_human_mode


parser = argparse.ArgumentParser(description='Doom Demo Recorder')
parser.add_argument('--env-name', default='DoomBasic-v0', metavar='ENV',
                    help='environment to train on (default: DoomBasic-v0)')
parser.add_argument('--num-eps', type=int, default='5', metavar='NE',
                    help='number of demo episodes to record')

if __name__ == '__main__':
    args = parser.parse_args()
    saver = TransitionSaver()

    for i in range(args.num_eps):
        env = gym.make('ppaquette/' + args.env_name)
        env = ToDiscrete("minimal")(env)
        env.unwrapped._mode = 'human'
        env.reset()

    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    saver.save('demos/' + args.env_name + '_demo_' + 'fs_'  + str(FRAMESKIP) + '_' + timestring + '.p')