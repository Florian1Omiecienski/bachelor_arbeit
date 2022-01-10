#!/usr/bin/python3
"""
This file holds a class for calculating the context encodings used for the link score (see Bachelor Thesis section 4.7).

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn

from .ffnn import FFNN
from .maximum import Maximum


class AffineMultiQueryHardAttentionEncoder(nn.Module):
    """
    This class is used for calculating one encoding for a bag of vectors using mutltiple query vectors.
    This class is to encode the context words of entity spans (see section 4.7 in the bachelor thesis).
    """
    def __init__(self, dimension, k, low_parameter=True):
        super(AffineMultiQueryHardAttentionEncoder, self).__init__()
        # Hyper-Parameters
        self.K = k
        self.dimension = dimension
        self.low_parameter = low_parameter
        # Components
        if self.low_parameter is True:
            self.affine = nn.Parameter(torch.empty(dimension))
            nn.init.constant_(self.affine, math.sqrt(2))
            
        else:
            self.affine = nn.Parameter(torch.empty(dimension,dimension))
            nn.init.xavier_uniform_(self.affine, gain=math.sqrt(2))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, queries, values, keys):
        N, Q = queries.shape
        M, K = keys.shape
        M, V = values.shape
        if self.low_parameter is True:
            weight_matrix = torch.diag(self.affine)
        else:
            weight_matrix = self.affine
        attention_scores = Maximum.max(torch.matmul(torch.matmul(queries,weight_matrix ), keys.transpose(0,1)), dim=0)
        #
        weights, indices = torch.topk(attention_scores, k=min(self.K,attention_scores.shape[0]) , dim=0)
        weights = self.softmax(weights)
        weights = weights.unsqueeze(-1)
        #
        weighted_values = weights*values[indices]
        #
        encoding = torch.sum(weighted_values, dim=0)
        return encoding, indices
