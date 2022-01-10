#!/usr/bin/python3
"""
This file holds a class for claculating the dot product.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import torch


class DotProduct(object):
    """
    A class for calculating the dot product or similar. Methods can be tracked by pytorch.
    """
    def simpel(x, y):
        return torch.matmul(x,y.transpose(0,1))
    
    def cossim(x, y, eps=1e-8):
        a = torch.linalg.norm(x, dim=1).unsqueeze(-1)
        b = torch.linalg.norm(y, dim=1).unsqueeze(0)
        den = a*b
        den[den==0] = eps
        num = torch.matmul(x,y.transpose(0,1))
        return num/den
    
    def angular(x, y):
        ndot = DotProduct.cossim(x, y)
        return torch.arccos(ndot)
