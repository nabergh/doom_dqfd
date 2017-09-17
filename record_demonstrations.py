import sys, gym
import ppaquette_gym_doom
# from doom_utils import ToDiscrete
from utils import ReplayMemory, PreprocessImage
from datetime import date
import time
import pickle

class TransitionSaver:
    def __init__(self):
        self.processor = PreprocessImage(None)
        self.memory = ReplayMemory()

    def new_episode(self, first_state):
        self.state = self.processor._observation(first_state)

    def add_transition(self, action, next_state, reward):
        next_state = self.processor._observation(next_state)
        self.memory.push(self.state, self.add_noop(action), next_state, reward)
        self.state = next_state
    
    def add_noop(self, actions):
        new_actions = actions
        new_actions.insert(1, 0)
        for action in actions:
            if action != 0:
                new_actions[0] = 0
                return new_actions

    def save(self, fname):
        with open(fname, 'wb') as memory_file:
            pickle.dump(self.memory, memory_file)


saver = TransitionSaver()
def _play_human_mode(self):
    state = self.game.get_state().image_buffer.copy()
    saver.new_episode(state)
    while not self.game.is_episode_finished():
        self.game.advance_action()
        img = self.game.get_state().image_buffer
        if img is not None:
            state = img.copy()
        action = self.game.get_last_action()
        reward = self.game.get_last_reward()
        saver.add_transition(action, state, reward)
        time.sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
    return

ppaquette_gym_doom.doom_env.DoomEnv._play_human_mode = _play_human_mode


DOOM_ENV = 'DoomBasic-v0'

for i in range(10):
    env = gym.make('ppaquette/' + DOOM_ENV)
    # env = ToDiscrete("minimal")(env)
    env.unwrapped._mode = 'human'
    env.reset()

timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
saver.save('DoomBasic_demo_' + timestring)