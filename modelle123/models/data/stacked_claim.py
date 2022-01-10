#!/usr/bin/python3
"""
This file holds the stacked claim class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import unicodedata


def normalize_unicode(string):
    norm = string.replace("\n", "")
    norm = norm.replace("\xa0", " ")
    return ascii(unicodedata.normalize('NFD', norm))


class StackedClaim(object):
    """
    The StackedClaim class holds a list of actors, a list of categories and span.
    This class holds gold and predicted data.
    """
    def __init__(self, anno_id=None, text=None, span=None, categories=None, actors=None, unstacked_claims=None, p_actors=None, p_unstacked_claims=None, i_categories=None):
        self.anno_id = anno_id
        self.text = text
        self.span = span
        self.unstacked_claims = unstacked_claims   # Orginial unstacked claims
        # Gold annotations
        self.categories = categories
        self.actors = actors
        # Predicted annotations
        self.p_actors = p_actors
        self.p_unstacked_claims = p_unstacked_claims
        # Vars for index mapping
        self.i_categories = i_categories
    
    def __str__(self, intend=""):
        s = ""
        s += intend+"<Claim-Id: {}\n".format(self.anno_id)
        s += intend+" Text: '{}',\n".format(normalize_unicode(self.text))
        s += intend+" Token-Span: {},\n".format(self.span)
        s += intend+" Categories: {},\n".format(str(self.categories))
        s += intend+" Actors: {},\n".format(tuple([str(a) for a in self.actors]) if self.actors is not None else None)
        return s
    
    def to_tuple(self):
        s = (self.anno_id,
             self.text,
             self.span,
             self.unstacked_claims,
             self.categories,
             self.actors,
             self.p_actors,
             self.p_unstacked_claims,
             self.i_categories)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
    