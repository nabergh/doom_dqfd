import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DQN(nn.Module):
    def __init__(self, dtype, input_shape, num_actions):
        super(DQN, self).__init__()
        self.dtype = dtype

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

class DQN_rnn(nn.Module):
    def __init__(self, dtype, input_shape, num_actions):
        super(DQN_rnn, self).__init__()
        self.dtype = dtype

        self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        conv_out_size = self._get_conv_output(input_shape)
        self.rnn_size = 256

        self.lin1 = nn.Linear(conv_out_size, 256)
        self.rnn = nn.GRUCell(256, self.rnn_size)
        self.lin2 = nn.Linear(self.rnn_size, 256)
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

    def forward(self, states, hx):
        x = self._forward_conv(states)

        # flattening each element in the batch
        x = x.view(states.size(0), -1)
        x = F.leaky_relu(self.lin1(x))
        hx = self.rnn(x, hx)
        x = F.leaky_relu(self.lin2(hx))
        return self.lin3(x), hx