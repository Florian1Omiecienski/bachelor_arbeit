#!/usr/bin/python3
"""
This file holds the token class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

class Token(object):
    """
    This class holds a tokens span and text.
    """
    def __init__(self, text=None, index=None, span=None, followed_by_space=None):
        # Basic information
        self.text = text
        self.index = index
        self.span = span
        self.followed_by_space = followed_by_space
        # Vars for index mapping
        self.i_fasttext = None
        self.i_chars = None
    
    def to_tuple(self):
        s = (self.text,
             self.index,
             self.span,
             self.followed_by_space,
             self.i_fasttext,
             self.i_chars)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
