#!/usr/bin/python3
"""
This file holds a highway neural network class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayLayer(nn.Module):
    def __init__(self, hidden_size):
        """
        One layer of a highway network.
        """
        super(HighwayLayer, self).__init__()
        #
        self.hidden_size = hidden_size
        #
        self.hidden_linear = nn.Linear(in_features=hidden_size,
                                       out_features=hidden_size)
        self.gate_linear = nn.Linear(in_features=hidden_size,
                                     out_features=hidden_size)
        self.activation = F.relu
        #
        self._init_()
    
    def _init_(self):
        std = std = 1/math.sqrt(self.hidden_size)
        torch.nn.init.xavier_uniform_(self.hidden_linear.weight.data, gain=math.sqrt(2))
        torch.nn.init.uniform_(self.hidden_linear.bias.data, -std, std)
        #
        torch.nn.init.xavier_uniform_(self.gate_linear.weight.data, gain=math.sqrt(2))
        torch.nn.init.uniform_(self.gate_linear.bias.data, -std, std)
    
    def forward(self, input_tensor):
        # Input-Size (N, D)
        x = input_tensor
        # Calculate transformed input
        x_t = self.activation(self.hidden_linear(x))
        # Calculte transform gates
        T_gate = torch.sigmoid(self.gate_linear(x))
        C_gate = 1-T_gate
        # Calculate output
        out = (x*C_gate) + (x_t*T_gate)
        return out