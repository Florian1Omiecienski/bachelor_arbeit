#!/usr/bin/python3
"""
This file holds a class for text span encoder used for entities and claims in model 1 and model 2.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn

from .ffnn import FFNN


class SingleQueryEncoder(nn.Module):
    """
    A pytroch class for calculation a represenation from a sequence of vectors.
    Used as claim-span-representations in model 1 & 2.
    """
    def __init__(self, dimension, hidden_size, num_layers, dropout_p):
        super(SingleQueryEncoder, self).__init__()
        # Hyper-Parameters
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        # Components
        self.ffnn = FFNN(input_size=self.dimension,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout_p=self.dropout_p,
                         projection_dim=None)
        self.query_vector = nn.Parameter(torch.empty(hidden_size, 1))
        a = 1/math.sqrt(self.hidden_size)
        nn.init.uniform_(self.query_vector, -a, a)
    
    def forward(self, keys, values):
        hidden_keys = self.ffnn(keys)
        attention_scores = torch.matmul(hidden_keys, self.query_vector)
        weighted_values = attention_scores*values
        encoding = torch.sum(weighted_values, dim=0)
        return encoding
