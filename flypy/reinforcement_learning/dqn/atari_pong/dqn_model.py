"""Neural Network Model"""

import torch
import torch.nn as nn
import numpy as np



class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # Convolutional layer from the Nature
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=5),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  nn.ReLU())

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential()
