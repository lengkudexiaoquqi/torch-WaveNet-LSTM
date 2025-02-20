import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from torchsummary import summary
import numpy as np
class LSTM(nn.Module):
    def __init__(self,hidden_size,predict_step):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=1,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, predict_step)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class RNN(nn.Module):
    def __init__(self,hidden_size,predict_step):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN(), it hardly learns
            input_size=1,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, predict_step)

    def forward(self, x):
        r_out ,h = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out


class BPNet(nn.Module):

    def __init__(self,window_size,hidden_size,predict_step):
        super(BPNet, self).__init__()
        self.Flatten=nn.Flatten()
        self.Linear1=nn.Linear(window_size,hidden_size)
        self.Sigmoid=nn.Sigmoid()
        self.Linear2=nn.Linear(hidden_size,predict_step)

    def forward(self, x):
        x=self.Flatten(x)
        x=self.Sigmoid(self.Linear1(x))
        y_pre=self.Linear2(x)
        return y_pre

