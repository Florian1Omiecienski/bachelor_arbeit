"""
Contains a helper function to count entity-word-cooccurence.
"""

import numpy as np


class Counter(object):
    """
    A helper class. Counts the co-occurence of entities and words. 
    Used in the EmbeddingBootstrapper.
    """
    def __init__(self, entity_map=None, word_map=None):
        self.word_map = word_map
        self.entity_map = entity_map
        self.map = dict()
    
    def __call__(self, a, b):
        return self.count(a, b)
    
    def count(self, a, b):
        """
        Increases count of (a, b)
        """
        try:
            e_dict = self.map[a]
            try:
                e_dict[b] +=1
            except KeyError:
                 e_dict[b] = 1
        except KeyError:
            self.map[a] = dict([(b,1), ])
        return None
    
    def to_numpy(self):
        """
        Returns a tripel (counts, entity_map, word_map).
        counts is a numpy-array of shape (num_entities, num_words) holding the counts.
        entity_map und word_map are dictionarys mapping the string-reps to indices.
        """
        ### Calculate word-counts
        # Convert to numpy-array and create index-mappings
        if self.entity_map is None:
            self.entity_map = dict([(l,i) for i,l in enumerate(self.map.keys())])
        else:
            for e in self.map.keys():
                if e not in self.entity_map:
                    self.entity_map[e] = len(self.entity_map)
        if self.word_map is None:
            all_words = set()
            for v in self.map.values():
                all_words.update(v.keys())
            self.word_map = dict([(l,i) for i,l in enumerate(all_words)])
        #
        n,m = len(self.entity_map),len(self.word_map)
        array = np.zeros((n,m))
        #
        for il, iv in self.entity_map.items():
            for jl, jv in self.word_map.items():
                try:
                    array[iv, jv] = self.map[il][jl]
                except KeyError:
                    pass
        #
        return array, self.entity_map, self.word_map
