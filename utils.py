import random
from collections import namedtuple

import torch
import torchvision.transforms as T
from PIL import Image
from gym.core import ObservationWrapper


# environment wrapper to process the input frames and turn them into pytorch tensors
class PreprocessImage(ObservationWrapper):
    def __init__(self, env, width=80, height=80):
        if env is not None:
            super(PreprocessImage, self).__init__(env)
        self.resize = T.Compose([T.ToPILImage(),
                    T.Scale((width,height), interpolation=Image.CUBIC),
                    T.ToTensor()])

# image transformations, to a 1x1x80x80 tensor
    def _observation(self, screen):
        screen = torch.from_numpy(screen)
        screen = screen.permute(2, 1, 0)
        screen = self.resize(screen)
        screen = screen.mean(0, keepdim=True)
        screen = screen.unsqueeze(0)
        return screen



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'n_reward'))

# object to hold all the transformations from which batches are sampled
class ReplayMemory(object):

    # capacity == -1 means unlimited capacity
    def __init__(self, capacity=-1):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        if len(self.memory) < self.capacity or self.capacity < 0:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = self.position + 1
        if self.capacity > 0:
            self.position = self.position % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class EpsGreedyPolicy(object):
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        eps_steps = max(eps_steps, 1)
        self.steps_done = 0

    # epsilon changes linearly from eps_start to eps_end within the first eps_steps number of steps
    def select_action(self, q_vals, env):
        sample = random.random()
        self.steps_done += 1
        if sample > min(self.steps_done / self.eps_steps, 1) * (self.eps_end - self.eps_start) + self.eps_start:
            return q_vals.max(1)[1].cpu()
        else:
            return torch.LongTensor([env.action_space.sample()])
