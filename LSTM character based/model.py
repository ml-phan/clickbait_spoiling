""" This file contains the main code for the LSTM class itself. """

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        """ Initialize Model with some hyperparameters. """

        super(LSTM, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # define main LSTM layers and fc layer for classification
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        # initialize hidden and cell states
        h0 = torch.rand(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.rand(self.num_layers, x.size(1), self.hidden_size).to(self.device)

        # send through lstm layer, drop first dimension and through fc layer
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        out = self.fc(out)

        return out
