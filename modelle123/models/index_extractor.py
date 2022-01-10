#!/usr/bin/python3
"""
This file holds the code for the IndexExtractor class.
This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""


import pickle
import gzip

from .data import StringMapper


class IndexExtractor(object):
    """
    This class maps words, chars, actor-ids, etc. to indices and stores the mapping.
    """
    def __init__(self, word_map, actor_map, category_map=None):
        self.word_mapper = StringMapper(map_=word_map, use_unk=True, freeze=True)
        self.actor_mapper = StringMapper(map_=actor_map, use_unk=False, freeze=True)
        self.char_mapper = StringMapper(use_unk=True)
        self.entity_mapper = StringMapper(use_unk=True)
        self.category_mapper = StringMapper(use_unk=True) if category_map is None else StringMapper(map_=category_map, use_unk=False, freeze=True)
        self.polarity_mapper = StringMapper(use_unk=False)
        self.distance_mapper = StringMapper(use_unk=True)
    
    def _extract_words(self, doc):
        for token in doc.tokens:
            token_idx = self.word_mapper.lookup(token.text.lower())
            token.i_fasttext = token_idx
        return doc
    
    def _extract_chars(self, doc):
        for token in doc.tokens:
            token.i_chars = []
            for char in token.text:
                char_idx = self.char_mapper.lookup(char)
                token.i_chars.append(char_idx)
            token.i_chars = tuple(token.i_chars)
        return doc
    
    def _extract_actors(self, doc):
        for claim in doc.claims:
            if claim.actor.is_nil is False:
                actor_wd = claim.actor.wikidata_id
                actor_idx = self.actor_mapper.lookup(actor_wd)
                claim.actor.i_wikidata_id = actor_idx
        return doc
    
    def _extract_entities(self, doc):
        for entity in doc.entities:
            # Map the entity-class to an index
            e_type = entity.entity_class
            e_type_idx = self.entity_mapper.lookup(e_type)
            entity.i_entity_class = e_type_idx
            # Map the distance_feature to index
            feature_idx = self.entity_mapper.lookup(str(entity.distance_feature))
            entity.i_distance_feature = feature_idx
        return doc
    
    def _extract_claims(self, doc):
        # Extract unstacked-claims
        for claim in doc.claims:
            # Map claim categories to indices
            claim.i_categories = []
            for category in claim.categories:
                category_idx = self.category_mapper.lookup(category)
                claim.i_categories.append(category_idx)
            claim.i_categories = tuple(sorted(claim.i_categories))
            # Map polarity to index
            polarity_idx = self.polarity_mapper.lookup(claim.polarity)
            claim.i_polarity = polarity_idx
        # Extract stacked-claims
        for claim in doc.stacked_claims:
            # Map claim categories to indices
            claim.i_categories = []
            for category in claim.categories:
                category_idx = self.category_mapper.lookup(category)
                claim.i_categories.append(category_idx)
            claim.i_categories = tuple(sorted(claim.i_categories))
        return doc
    
    def _extract_distances(self, doc):
        new_rows = []
        for row in doc.distances:
            new_row = []
            for dist in row:
                dist_idx = self.distance_mapper.lookup(str(dist))
                new_row.append(dist_idx)
            new_rows.append(tuple(new_row))
        doc.i_distances = tuple(new_rows)
        return doc
    
    def extract_all_gold(self, documents):
        """
        Extracts all indices for all documents in the specified list.
        Extraction goes from data to indices.
        """
        for doc in documents:
            self._extract_words(doc)
            self._extract_chars(doc)
            self._extract_actors(doc)
            self._extract_entities(doc)
            self._extract_claims(doc)
            self._extract_distances(doc)
        return documents
    
    def freeze(self, value=True):
        """
        Freeze the maps. No new indices can be created after calling this methods. UNK will be returned on new data.
        """
        self.word_mapper.freeze(value)
        self.actor_mapper.freeze(value)
        self.char_mapper.freeze(value)
        self.entity_mapper.freeze(value)
        self.category_mapper.freeze(value)
        self.polarity_mapper.freeze(value)
        self.distance_mapper.freeze(value)
    
    def _inverse_extract_actors(self, doc):
        if doc.p_actors is not None:
            for actor in doc.p_actors:
                p_wikidata_id = self.actor_mapper.inverse_lookup(actor.pi_wikidata_id)
                actor.p_wikidata_id = p_wikidata_id
    
    def _inverse_extract_stacked_claims(self, doc):
        if doc.p_stacked_claims is not None:
            for claim in doc.p_stacked_claims:
                for actor in claim.p_actors:
                    if actor.is_nil is False:
                        p_wikidata_id = self.actor_mapper.inverse_lookup(actor.pi_wikidata_id)
                        actor.p_wikidata_id = p_wikidata_id
    
    def _inverse_extract_unstacked_claims(self, doc):
        if doc.p_claims is not None:
            for claim in doc.p_claims:
                #
                actor_idx = claim.p_actor.pi_wikidata_id
                claim.p_actor.p_wikidata_id  = self.actor_mapper.inverse_lookup(actor_idx)
                #
                #pol = self.polarity_mapper.inverse_lookup(claim.pi_polarity)
                #claim.p_polarity = pol
                #
                claim.p_categories = []
                for cidx in claim.pi_categories:
                    cat = self.category_mapper.inverse_lookup(cidx)
                    claim.p_categories.append(cat)
                claim.p_categories = tuple(sorted(claim.p_categories, key=lambda x:str(x)))
    
    def inverse_extract_all_predictions(self, documents):
        """
        Extract data from indices. Use this on predicted files.
        Does not recreate actor names only wikidata ids.
        """
        for doc in documents:
            self._inverse_extract_actors(doc)
            self._inverse_extract_stacked_claims(doc)
            self._inverse_extract_unstacked_claims(doc)
    
    def word_map(self):
        """
        Returns a dictionary
        """
        return self.word_mapper.map
    
    def actor_map(self):
        return self.actor_mapper.map
    
    def char_map(self):
        return self.char_mapper.map
    
    def entity_map(self):
        return self.entity_mapper.map
    
    def category_map(self):
        return self.category_mapper.map
    
    def polarity_map(self):
        return self.polarity_mapper.map
    
    def distance_map(self):
        return self.distance_mapper.map
    
    def save(self, path):
        """
        Stores the indexExtractor to the specified path
        """
        with gzip.open(path,'wb') as ofile:
            pickle.dump(self, ofile)
        return None
    
    @staticmethod
    def load(path):
        """
        Loads an indexExtractor from the specified path
        """
        with gzip.open(path,'rb') as ifile:
            extractor = pickle.load(ifile)
        return extractor