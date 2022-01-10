#!/usr/bin/python3
"""
This file holds the claim class.
All methods outside the Claim-Class are helper methods.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import unicodedata
import re

from .actor import Actor


def normalize_unicode(string):
    norm = string.replace("\n", "")
    norm = norm.replace("\xa0", " ")
    return ascii(unicodedata.normalize('NFD', norm))


def argmin(list_):
    i,_ = min(enumerate(list_), key=lambda x:x[1])
    return i


def find_actor_span(mention_data, claim_char_span, doc_text):
    actor_spans = []
    in_text = mention_data["in text"].strip()
    iterator = re.finditer("(^|\s)({})(\s|\W|$)".format(in_text), doc_text[claim_char_span[0]:claim_char_span[1]])
    res = []
    for e in iterator:
        res.append(e)
    if len(res) == 1:
        span = res[0].span(2)
        span = (span[0]+claim_char_span[0],span[1]+claim_char_span[0])
        assert(doc_text[span[0]:span[1]] == in_text)
        actor_spans.append(span)
    return actor_spans


def process_orginial_annotations(anno_id, actor_name, claim_char_span, doc_text, orginial_annotations):
    actor_spans = []
    # Find original annotation of this claim span
    og_anno = None
    for oa in orginial_annotations:
        if oa["id"]==anno_id:
            og_anno = oa
            break
    og_anno = og_anno["data"]
    # Check if actor-information is available
    if "actorclusters" in og_anno:
        # If only one actor is annotated for this claim
        # it mus be the wanted actor
        if len(og_anno["actorclusters"]) == 1:
            ac = og_anno["actorclusters"][0]
            if ac["outside"] is False:
                for mention_data in ac["inside"]:
                    found_span = find_actor_span(mention_data, claim_char_span, doc_text)
                    actor_spans.extend(found_span)
        # If more than one actor is annotated for this claim ...
        else:
            for ac in og_anno["actorclusters"]:
                if ac["outside"] is False:
                    for mention_data in ac["inside"]:
                        # then search for an actor_span with matching name
                        # Note: Only actors that are realized via there normalized name,
                        #       can be found here.
                        if actor_name==mention_data["actor"]:
                            found_span = find_actor_span(mention_data, claim_char_span, doc_text)
                            actor_spans.extend(found_span)
    return actor_spans


def from_char_to_token(char_start, char_end, token_boundaries):
    # Find start- and end-token of claim
    char_diff_start = [char_start-s for s,_ in token_boundaries]
    char_diff_end = [char_end-e for _,e in token_boundaries]
    start_idx = argmin([abs(e) for e in char_diff_start])
    end_idx = argmin([abs(e) for e in char_diff_end])+1
    if start_idx == end_idx:
        print("Error in claim.py; from_char_to_token()")
        return None
    return start_idx, end_idx


def token_indices_to_entitiy(token_start, token_end, entities):
    for i in range(len(entities)):
        e = entities[i]
        if ((token_start<=e.span[0]) and (e.span[0]<token_end)) or ((token_start<e.span[1]) and (e.span[1]<=token_end)):  # partital matches are counted
            return i
    return None


class Claim(object):
    """
    The Claim class holds an actor, a list of categories and a polarity.
    This class holds gold and predicted data.
    """
    def __init__(self, anno_id=None, text=None, span=None, categories=None, actor=None, polarity=None, p_actor=None, p_polarity=None, p_categories=None, 
                i_categories=None, pi_categories=None, i_polarity=None, pi_polarity=None):
        self.anno_id = anno_id
        self.text = text
        self.span = span
        # Gold annotations
        self.categories = categories
        self.actor = actor
        self.polarity = polarity
        # Predicted annotations
        self.p_actor = p_actor
        self.p_polarity = p_polarity
        self.p_categories = p_categories
        # Vars for index-mappings
        self.i_categories = i_categories
        self.pi_categories = pi_categories
        self.i_polarity = i_polarity
        self.pi_polarity = pi_polarity
    
    def __str__(self, intend=""):
        s = ""
        s += intend+"<Claim-Id: {}\n".format(self.anno_id)
        s += intend+" Text: '{}',\n".format(normalize_unicode(self.text))
        s += intend+" Token-Span: {},\n".format(self.span)
        s += intend+" Categories: {},\n".format(str(self.categories))
        s += intend+" Actor: {},\n".format(str(self.actor))
        s += intend+" Polarity: {}>".format(str(self.polarity))
        return s
    
    def to_tuple(self):
        s = (self.anno_id,
             self.text,
             self.span,
             self.categories,
             self.actor,
             self.polarity,
             self.p_actor,
             self.p_polarity,
             self.i_categories,
             self.i_polarity,
             self.pi_polarity)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
    
    @staticmethod
    def from_mardy_data(doc_text, claim_annotation, orginial_annotations, token_boundaries, entities, actor_mapping, tokens_debug):
        """
        Loads a claim from the annotations. This methods is used by the datahandler to load the debatenet documents.
        Returns a claim object which holds gold data.
        """
        # Basic claim attributes
        claim_anno_id = claim_annotation["anno_id"]
        claim_text = claim_annotation["quote"]
        actor_name = claim_annotation["name"].strip(" ")
        polarity = claim_annotation["cpos"]
        labels = claim_annotation["claimvalues"]
        # Sanity check data
        for l in labels:
            try:
                int(l)
            except ValueError:
                print("Malformed Claim-Label ({})".format(l))
                print("    --> Solution: Remove-Label from claim")
                labels.remove(l)
        labels = [int(l) for l in labels]
        # Correct char offset of claim
        char_start = claim_annotation["begin"]
        char_end = claim_annotation["end"]
        while doc_text[char_start] in [" ", "\n"]:
            char_start+=1
        while doc_text[char_end-1] in [" ", "\n"]:
            char_end-=1
        # Map claim char offsets to token offsets
        res = from_char_to_token(char_start, char_end, token_boundaries)
        if res is None:
            raise Exception("claim.py: Error-1")
        claim_span = res
        # If available and unambiguous, read the char-span in which the actor is realized
        actor_spans = process_orginial_annotations(claim_anno_id,
                                                   actor_name,
                                                   (char_start, char_end),
                                                   doc_text,
                                                   orginial_annotations)
        # For all available actor char spans ...
        for i in range(len(actor_spans)):
            aspan = actor_spans[i]
            # Map char-offsets to token-offsets
            res = from_char_to_token(aspan[0], aspan[1], token_boundaries)
            if res is None:
                raise Exception("claim.py: Error-2")
            # Map token offsets to an entity-span (if one exists)
            actor_spans[i] = token_indices_to_entitiy(res[0], res[1], entities)
        while None in actor_spans:
            actor_spans.remove(None)
        # Map actor name to wikidata-id if possible
        actor_wd_name = None
        if actor_name in actor_mapping:
            actor_wd_name = actor_mapping[actor_name]
        # Create Actor-Object
        actor_spans = tuple(sorted(actor_spans))
        actor = Actor(normal_name=actor_name,
                      wikidata_id=actor_wd_name,
                      spans=actor_spans,
                      is_nil=(actor_wd_name is None))
        # Create Claim-Object
        categories = tuple(sorted(labels))
        actor_spans = tuple(sorted(actor_spans))
        assert(len(categories)>0)
        new_claim = Claim(anno_id=claim_anno_id,
                          text=claim_text,
                          span=claim_span,
                          categories=categories,
                          polarity=polarity,
                          actor=actor)
        return new_claim
