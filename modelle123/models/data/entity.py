#!/usr/bin/python3
"""
This file holds the entity class.
All methods outside the Entity-Class are helper methods.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import unicodedata


def normalize_unicode(string):
    norm = string.replace("\n", "")
    norm = norm.replace("\xa0", " ")
    return ascii(unicodedata.normalize('NFC', norm))


def argmin(list_):
    i,_ = min(enumerate(list_), key=lambda x:x[1])
    return i


def from_char_to_token(char_start, char_end, token_boundaries):
    # Find start- and end-token of claim
    char_diff_start = [char_start-s for s,_ in token_boundaries]
    char_diff_end = [char_end-e for _,e in token_boundaries]
    start_idx = argmin([abs(e) for e in char_diff_start])
    end_idx = argmin([abs(e) for e in char_diff_end])+1
    flag1,flag2=False,False
    if char_diff_start[start_idx] < 0:
        start_idx -= 1
    if char_diff_end[end_idx-1] > 0:  # some nes are sub-token level
        end_idx += 1
    if start_idx == end_idx:
        print("Error in entity.py; from_char_to_token()")
        return None
    return start_idx, end_idx


class Entity(object):
    """
    The EntityClass holds a text-span and an entity-class.
    This class holds gold data.
    """
    def __init__(self, text=None, entity_class=None, span=None, is_auto=None):
        # Gold-Annotations
        self.text = text                    # str
        self.entity_class = entity_class    # str
        self.span = span                    # tuple(int, int)
        self.is_auto = is_auto              # bool()
        # Additional feature that can be generated
        self.distance_feature = None
        # Vars for index-mappings
        self.i_entity_class = None          # int
        self.i_distance_feature = None      # int
    
    def __str__(self):
        s = "<Text: '{}', Class: {}, Dist2NextClaim: {}, Token-Span: {}, is_auto={}>".format(self.text, self.entity_class, self.distance_feature, self.span, self.is_auto)
        return s
    
    def to_tuple(self):
        s = (self.text,
             self.entity_class,
             self.span,
             self.is_auto,
             self.distance_feature,
             self.i_entity_class,
             self.i_distance_feature)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
    
    @staticmethod
    def from_mardy_data(doc_text, entity_annotation, token_boundaries, is_automatically_detected):
        # Basic entity attributes
        entity_text = entity_annotation["quote"]
        entity_type = entity_annotation["entity"]
        char_start = entity_annotation["begin"]
        char_end = entity_annotation["end"]
        # Correct char offsets if needed
        while doc_text[char_start] in [" ", "\n"]:
            char_start += 1
        while doc_text[char_end-1] in [" ", "\n"]:
            char_end -= 1
        # Sanity-Check if spans are set correctly
        if normalize_unicode(entity_text) != normalize_unicode(doc_text[char_start:char_end]):
            print("Maleformed entity-annotation: Anno=[{}], Doc-Text=[{}]".format(normalize_unicode(entity_text) , normalize_unicode(doc_text[char_start:char_end])))
            print("    --> Solution: Remove annotation from data")
            return None
        # Find token offset of entity-span
        entity_span = from_char_to_token(char_start, char_end,  token_boundaries)
        if entity_span is None:
            print("Error-1")
            return None
        # Create new Entity
        new_entity =  Entity(text=entity_text,
                             entity_class=entity_type,
                             span=entity_span,
                             is_auto=is_automatically_detected)
        return new_entity
