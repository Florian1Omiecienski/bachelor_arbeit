#!/usr/bin/python3
"""
This file holds a class for text span encoder used for entities and claims in model 3.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""


import math
import torch
import torch.nn as nn

from .ffnn import FFNN


class MultiQueryEncoder(nn.Module):
    """
    A pytroch class for calculation a some represenations from a sequence of vectors and some query-vectors.
    Used as claim-span-representations in model 3.
    """
    def __init__(self, dimension, num_queries, hidden_size, num_layers, dropout_p):
        super(MultiQueryEncoder, self).__init__()
        # Hyper-Parameters
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dropout_p = dropout_p
        
        # Components
        self.ffnn = FFNN(input_size=self.dimension,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout_p=self.dropout_p,
                         projection_dim=None)
        self.hidden_dropout = nn.Dropout(self.dropout_p)
        self.query_vectors = nn.Parameter(torch.empty(hidden_size, num_queries))
        a = 1/math.sqrt(self.hidden_size)
        nn.init.uniform_(self.query_vectors, -a, a)
    
    def forward(self, query_ids, keys, values):
        hidden_keys = self.ffnn(keys)
        hidden_keys = self.hidden_dropout(hidden_keys)
        attention_scores = torch.matmul(hidden_keys, self.query_vectors[:, query_ids])
        weighted_values = attention_scores.unsqueeze(-1)*values.unsqueeze(1)
        encodings = torch.sum(weighted_values, dim=0)
        return encodings
