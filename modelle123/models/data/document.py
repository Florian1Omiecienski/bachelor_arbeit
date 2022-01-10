#!/usr/bin/python3
"""
This file holds the document class.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

from .claim import Claim
from .stacked_claim import StackedClaim
from .entity import Entity
from .my_token import Token




class Document(object):
    """
    This class holds all data for and from a debateNet document.
    This class holds a actor set on doc-level.
    This class holds a list of claims (unstacked).
    This class holds a list of stacked claims.
    Also this class holds the predicted data.
    """
    def __init__(self, doc_id, text, tokens, entities, claims, stacked_claims, actors, distances):
        # Document attributes and data
        self.doc_id = doc_id
        self.text = text
        self.tokens = tokens
        # Gold-Annotations
        self.entities = entities
        self.claims =  claims
        #
        self.stacked_claims = stacked_claims
        self.actors = actors
        #
        self.distances = distances
        #
        self.i_distances = None
        # Predicted annotations
        self.p_actors = None
        self.p_stacked_claims = None
        self.p_claims =  None
    
    def __str__(self):
        s = ""
        s += "Doc-ID: {}\n".format(self.doc_id)
        s += "Doc-Text: '{}'\n".format(self.text[:10]+" [...] "+self.text[-10:])
        s += "#Tokens: {}\n".format(len(self.tokens))
        s += "Entities:\n\t"
        s += "\n\t".join([str(e) for e in self.entities])+"\n"
        s += "Claims:\n"
        s += "\n".join([c.__str__(intend="\t") for c in self.stacked_claims])+"\n"
        s += "Actors:\n\t"
        s += "\n\t".join([str(a) for a in self.actors])+"\n"
        return s
    
    def to_tuple(self):
        s = (self.doc_id,
             self.text,
             self.tokens,
             self.entities,
             self.claims,
             self.stacked_claims,
             self.actors,
             self.distances,
             self.p_actors,
             self.p_stacked_claims,
             self.p_claims)
        return s
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, o):
        if type(o) is not type(self):
            return None
        flag = self.to_tuple() == o.to_tuple()
        return flag
    
    @staticmethod
    def from_mardy_data(doc_id, article, claim_annotations, original_annotations, entity_annotations, actor_mapping):
        """
        Loads a document from the annotations. This methods is used by the datahandler to load the debatenet documents.
        Returns a document which holds all gold data.
        """
        # Raw text from document
        text = article["text"]
        # Tokenize text by given sentence and token boundaries
        tokens = Document._apply_segmentation(text, article["sentences"], article["tokens"])
        # Create a list, mapping tokens to char-offsets in the raw text
        # This is not from article-anno because some offsets get corrected during tokenization
        token_boundaries = [(t.span[0], t.span[1]) for t in tokens]
        # Create Entity-Objects for all automatically/manually detected entities
        entities = Document._create_entities(text, article, entity_annotations, token_boundaries)
        # Resolving double annotations of one text-span
        entities = Document._clean_entities(entities)
        # Create Claim-Objects
        claims = Document._create_claims(text, tokens, entities, claim_annotations, original_annotations, token_boundaries, actor_mapping)
        # Create Document-wide actor set
        doc_wise_actors = set()
        for claim in claims:
            doc_wise_actors.add(claim.actor)
        doc_wise_actors = tuple(sorted(doc_wise_actors, key=lambda x:x.normal_name))
        # Create Stacked Version of claims
        stacked_claims = Document._stack_claims(claims)
        # Create Distance-Features for Claims and Entities
        distances = Document._create_distances(entities, claims)
        # Create Distance-Features for Entities
        for i in range(len(entities)):
            entity = entities[i]
            idx = Document._argmin([abs(dist) for dist in distances[i]])
            entity.distance_feature = distances[i][idx]
        # Create new document
        new_doc = Document(doc_id=doc_id,
                           text=text,
                           tokens=tuple(tokens),
                           claims=claims,
                           entities=entities,
                           stacked_claims=stacked_claims,
                           actors=doc_wise_actors,
                           distances=distances)
        return new_doc
    
    @staticmethod
    def _argmin(list_):
        i,_ = min(enumerate(list_), key=lambda x:x[1])
        return i
    
    @staticmethod
    def _apply_segmentation(doc_text, sentence_bounds, token_bounds):
        """
        Read tokenization from gold-data
        """
        all_tokens = []
        abs_idx = 0
        for i in range(len(sentence_bounds)):
            ss = int(sentence_bounds[i]["begin"])
            se = int(sentence_bounds[i]["end"])
            for j in range(len(token_bounds)):
                ts = int(token_bounds[j]["begin"])
                te = int(token_bounds[j]["end"])
                if ts < ss:
                    continue
                if (i+1 < len(sentence_bounds)) and (ts >= sentence_bounds[i+1]["begin"]):
                    break
                token_text = doc_text[ts:te]
                followed_by_space = True if doc_text[te] in [" ", "\n"] else False
                new_token = Token(text=token_text, index=abs_idx,  span=(ts,te), followed_by_space=followed_by_space)
                all_tokens.append(new_token)
                abs_idx += 1
        ##
        return all_tokens
    
    @staticmethod
    def _create_entities(doc_text, article_annotation, entity_annotations, token_boundaries):
        """
        Create entities from gold-annotation
        """
        # Create Entity-Objects for manually added entities
        entities = []
        for mea in entity_annotations:
            new_entity = Entity.from_mardy_data(doc_text, mea, token_boundaries, is_automatically_detected=False)
            if new_entity is None:
                continue
            entities.append(new_entity)
        # Rename automatically added entity annotations
        for ne in article_annotation["named_entities"]:
            type_new = ne["category"].split("-")[1]
            ne["entity"] = type_new
            del ne["category"]
            ne["quote"] = ne["text"]
            del ne["text"]
        # Create Entity-Objects for automatically detected NER-entities
        for mea in article_annotation["named_entities"]:
            new_entity = Entity.from_mardy_data(doc_text, mea, token_boundaries, is_automatically_detected=True)
            # If entity-cretion fails (eg. sanity check failed), then ignore it
            if new_entity is None:
                continue
            ## Use only automatically detected named-entities with PERS- or ORG-Class
            if new_entity.entity_class not in ["PER", "ORG"]:
                continue
            entities.append(new_entity)
        entities = tuple(sorted(entities, key=lambda x:x.span))
        return entities
    
    @staticmethod
    def _clean_entities(entities):
        """
        Remove situations where automatically and manual added entity-annotations conflict.
        """
        # Sort entities by there token span
        ents_by_span = dict()
        for e in entities:
            if e.span not in ents_by_span:
                ents_by_span[e.span] = set()
            ents_by_span[e.span].add(e)
        # If conflicting annotations of a text span are found
        # then try to solve the conflict ..
        for k,v in ents_by_span.items():
            # If a manual annotation is available
            # then remove all automatically created annotations
            if len(v) > 1:
                man_flag = False
                for e in v:
                    if e.is_auto is False:
                        man_flag = True
                        break
                if man_flag is True:
                    for e in set(v):
                        if e.is_auto is True:
                            v.remove(e)
            # If conflict is only in the the text-field
            # then select one annotation (it doesnt matter which, conflict due to erroneous annotation)
            # Note: If conflict is in entity-class, both annotations are kept
            #       to represent the conflict (assume intent of annotater)
            #       E.g. "Grünen-Cheffin" can act as PER or can represent an ORG
            if len(v) > 1:
                text_flags = []
                class_flags = []
                lv = list(v)
                for i in range(len(v)-1):
                    text_flags.append(lv[i].text==lv[i+1].text)
                    class_flags.append(lv[i].entity_class==lv[i+1].entity_class)
                text_flags = all(text_flags)
                class_flags = all(class_flags)
                if class_flags is True:
                    for i in range(1, len(lv)):
                        v.remove(lv[i])
        #
        entities = []
        for v in ents_by_span.values():
            entities.extend(v)
        #
        entities = tuple(sorted(entities, key=lambda x:x.span))
        return entities
    
    @staticmethod
    def _create_claims(doc_text, tokens, entities, claim_annotations, original_annotations, token_boundaries, actor_mapping):
        """
        Create claim objects from gold
        """
        claims = []
        for ca in claim_annotations:
            new_claim = Claim.from_mardy_data(doc_text, ca, original_annotations, token_boundaries, entities, actor_mapping, tokens)
            if new_claim is None:
                continue
            claims.append(new_claim)
        claims = tuple(sorted(claims, key=lambda x:x.span))
        return claims
    
    @staticmethod
    def _stack_claims(unstacked_claims):
        """
        stack claims by claim span
        """
        #
        claims_by_span = dict()
        for claim in unstacked_claims:
            if claim.span not in claims_by_span:
                claims_by_span[claim.span] = []
            claims_by_span[claim.span].append(claim)
        #
        stacked_claims = []
        for _, claims in claims_by_span.items():
            all_actors = set([c.actor for c in claims])
            all_categories = set()
            for c in claims:
                all_categories.update(c.categories)
            #
            all_actors = tuple(sorted(all_actors, key=lambda x:x.normal_name))
            all_categories = tuple(sorted(all_categories))
            unstacked_claims = tuple(sorted(claims, key=lambda x:x.span))
            stack_claim = StackedClaim(anno_id=claims[0].anno_id,
                                       span=claims[0].span,
                                       text=claims[0].text,
                                       actors=all_actors,
                                       categories=all_categories,
                                       unstacked_claims=unstacked_claims)
            stacked_claims.append(stack_claim)
        stacked_claims = tuple(sorted(stacked_claims, key=lambda x:x.span))
        return stacked_claims
    
    @staticmethod
    def _bin(integer, bins, keep_sign=True):
        """
        map an integer to a bin.
        bins are defined as a list of integers.
        """
        unsiged = abs(integer)
        sign = 1 if integer>=0 else -1
        # find bin
        return_val = None
        for i in range(len(bins)-1):
            lower_bin = bins[i]
            upper_bin = bins[i+1]
            if (lower_bin<=unsiged) and (unsiged<upper_bin):
                return_val = lower_bin
                break
        if return_val is None:
            return_val = bins[-1]
        # restore sign
        if keep_sign is True:
            return_val *= sign
        return return_val
    
    @staticmethod
    def _create_distances(entities, claims, bins=(0,1,2,3,4,8,16,32,64)):
        """
        Creates a table of distances between all entities and all claims
        """
        N = len(entities)
        M = len(claims)
        dist_tabel = list()
        for i in range(N):
            entity = entities[i]
            row = []
            for j in range(M):
                claim = claims[j]
                if entity.span[1] < claim.span[0]:  # entity is left of claim
                    dist = +(claim.span[0] - entity.span[1])
                elif claim.span[1] < entity.span[0]:  # entity is right of claim
                    dist = -(entity.span[0]-claim.span[1])
                else:  # entity-span is inside or intersecting the claim-span
                    dist = 0
                distance = Document._bin(dist, bins=bins)
                row.append(distance)
            dist_tabel.append(tuple(row))
        dist_tabel = tuple(dist_tabel)
        return dist_tabel

