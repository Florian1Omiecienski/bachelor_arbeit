#!/usr/bin/python3
"""
This file holds the string mapper class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

class StringMapper(object):
    """
    This class maps strings to indices. 
    New objects are assigned to a new index. 
    If the mapper is frozen, an UNK-index (0) is returned on unseen strings.
    """
    def __init__(self, map_=None, freeze=False, use_unk=True):
        self.use_unk = use_unk
        self.frozen = freeze
        self.idx = 0
        if map_ is None:
            init = []
            init_i = []
            if self.use_unk is True:
                init.append(("<UNK>",0))
                init_i.append((0,"<UNK>"))
                self.idx = 1
            self.map = dict(init)
            self.i_map = dict(init_i)
        else:
            assert( (("<UNK>" in map_) and (map_["<UNK>"]==0)) or ("<UNK>" not in map_) )
            self.map = map_
            self.i_map = dict([(i,v) for v,i in map_.items()])
            self.idx = len(map_)
    
    def lookup(self, string):
        """
        Lookup the string. Returns an integer.
        """
        if self.frozen is True:
            try:
                return self.map[string]
            except KeyError as e:
                if self.use_unk is True:
                    return self.map["<UNK>"]
                raise e
        else:
            try:
                return self.map[string]
            except KeyError:
                self.map[string] = self.idx
                self.i_map[self.idx] = string
                self.idx += 1
                return self.map[string]
    
    def inverse_lookup(self, idx):
        """
        Find the string for the given index.
        Returns a string
        """
        return self.i_map[idx]
    
    def freeze(self, value=True):
        """
        Make the stringMapper return UNK on unseen strings.
        """
        self.frozen = value
