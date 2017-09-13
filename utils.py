from collections import namedtuple

import torch
import torchvision.transforms as T
from PIL import Image
from gym.core import ObservationWrapper

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, width=80, height=80):
        if env is not None:
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
        return screen


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    # capacity == -1 means unlimited capacity
    def __init__(self, capacity=-1):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity or self.capacity < 0:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = self.position + 1
        if self.capacity > 0:
            self.position = self.position % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)