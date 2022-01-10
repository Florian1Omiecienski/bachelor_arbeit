#!/usr/bin/python3
"""
This file holds feedforward neural net class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    """
    This is a pytorch class.
    This is a feedforward neural net class using ReLU as activation.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout_p, projection_dim=None):
        super(FFNN, self).__init__()
        #
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.dropout_p = dropout_p
        #
        dropout_layer = nn.Dropout(self.dropout_p)
        relu_layer = nn.ReLU()
        sequence = [nn.Linear(in_features=self.input_size,
                              out_features=self.hidden_size),
                    relu_layer,]

        for i in range(num_layers-1):  
            sequence.append(dropout_layer)
            sequence.append(nn.Linear(in_features=self.hidden_size,
                                      out_features=self.hidden_size))
            sequence.append(relu_layer)
        if self.projection_dim is not None:
            sequence.append(dropout_layer)
            sequence.append(nn.Linear(in_features=self.hidden_size,
                                      out_features=self.projection_dim))
        self.sequence = nn.Sequential(*sequence)
        #
        std = 1/math.sqrt(self.hidden_size)
        for m in sequence:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data, gain=math.sqrt(2))
                torch.nn.init.uniform_(m.bias.data, -std, std)
    
    def forward(self, input_tensor):
        L, D = input_tensor.shape
        return self.sequence(input_tensor)
