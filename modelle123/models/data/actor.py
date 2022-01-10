#!/usr/bin/python3
"""
This file holds the actor class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

class Actor(object):
    """
    The actor class holds a normalized name, a wikidata-id, a flag indicating a nil-actor and sometimes a textspan
    This class holds gold and predicted data.
    """
    def __init__(self, normal_name=None, wikidata_id=None, is_nil=None, spans=None, 
                 p_normal_name=None, p_wikidata_id=None, p_spans=None, i_wikidata_id=None, pi_wikidata_id=None):
        # Gold-Annotations
        self.normal_name = normal_name  # str
        self.wikidata_id = wikidata_id  # str
        self.is_nil = is_nil            # bool
        self.spans = spans              # list([tuple(int, int), ...])
        # Predicted-Annotations
        self.p_normal_name = p_normal_name
        self.p_wikidata_id = p_wikidata_id
        self.p_spans = p_spans
        # Vars for index-mappings
        self.i_wikidata_id = i_wikidata_id
        self.pi_wikidata_id = pi_wikidata_id
    
    def to_tuple(self):
        s = (self.normal_name,
             self.wikidata_id,
             self.is_nil,
             self.spans,
             self.p_normal_name,
             self.p_wikidata_id,
             self.p_spans,
             self.i_wikidata_id,
             self.pi_wikidata_id)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
    
    def __str__(self):
        s = "<{} ({}), is_nil={}, spans={}, i_wikidata={}>".format(self.normal_name, self.wikidata_id, self.is_nil, str(self.spans), self. i_wikidata_id)
        return s
