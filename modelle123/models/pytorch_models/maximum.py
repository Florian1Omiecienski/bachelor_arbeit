#!/usr/bin/python3
"""
This file holds a class for calculating the maximum.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import torch


class Maximum(object):
    """
    Provides two maximums functions for pytorch.
    """
    def max(x, dim=-1):
        return torch.max(x,dim=dim).values
    
    def lse(x, dim=-1):
        return torch.logsumexp(x, dim=dim)
