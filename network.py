import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import CellGRU, CellLSTM

class Network(nn.Module):
    def __init__(self, in_features=1, out_features=1, rnn_units=32, rnn_cell='lstm', sigmoid=False):
        super(Network, self).__init__()
        self.sigmoid = sigmoid
        self.fc1 = nn.Linear(in_features=in_features, out_features=10)
        self.fc2 = nn.Linear(in_features=rnn_units, out_features=out_features)
        # self.relu = nn.ReLU()

        if(rnn_cell == 'lstm'):
            self.rnn = CellLSTM(10, out_features, units=rnn_units)
        elif(rnn_cell == 'gru'):
            self.rnn = CellGRU(10, out_features, units=rnn_units)

        if(self.sigmoid):
            self.sigmoid = nn.Sigmoid()

    def reset_hidden_state(self):
        self.rnn.reset_hidden_state()

    def forward(self, inputs):
        x    = self.fc1(inputs)
        c, h = self.rnn(x)

        y = self.fc2(h)

        return y, h

