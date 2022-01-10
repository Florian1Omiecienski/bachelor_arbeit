#!/usr/bin/python3
"""
This file holds a class for the third model.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn

from .ffnn import FFNN
from .word_encoder import WordEncoder
from .single_query_encoder import SingleQueryEncoder
from .multi_query_encoder import MultiQueryEncoder
from .hard_attention import AffineMultiQueryHardAttentionEncoder
from .dot_product import DotProduct
from .maximum import Maximum


class Model3(nn.Module):
    """
    This class provides the third model form the bachelor thesis (section 4.9)
    """
    def __init__(self, fast_text_matrix, actor_matrix, char_vocab_size, num_efeatures, num_cfeatures, num_dfeatures, char_pad_index,
                 efeature_embedding_dim, cfeature_embedding_dim, dfeature_embedding_dim,
                 char_embedding_dim, filters_per_channel, kernels, num_lstm_layers,
                 use_context, context_size, K_context, 
                 ffnn_hidden_dim, num_ffnn_layers, embedding_dropout_p, hidden_dropout_p):
        super(Model3, self).__init__()
        # Hyper-Parameters (Input)
        self.word_vocab_size = fast_text_matrix.shape[0]
        self.word_embedding_dim = fast_text_matrix.shape[1]
        
        self.num_efeatures = num_efeatures   # Number of entity-feature-embeddings
        self.num_cfeatures = num_cfeatures   # Number of claim-feature-embeddings
        self.num_dfeatures = num_dfeatures
        
        self.char_vocab_size = char_vocab_size
        self.char_pad_index = char_pad_index
        
        self.efeature_embedding_dim = efeature_embedding_dim
        self.cfeature_embedding_dim = cfeature_embedding_dim
        self.dfeature_embedding_dim = dfeature_embedding_dim
        
        self.use_context = use_context
        self.context_size = context_size
        self.K_context = K_context
        
        self.char_embedding_dim = char_embedding_dim
        self.filters_per_channel = filters_per_channel
        self.kernels = kernels
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = self.word_embedding_dim/2
        
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.num_ffnn_layers = num_ffnn_layers
        
        self.dot_product = DotProduct.simpel
        self.aggregate = Maximum.lse
        
        self.embedding_dropout_p = embedding_dropout_p
        self.hidden_dropout_p = hidden_dropout_p
        # Pretrained-Embeddings (Input)
        self.actor_embedding = nn.Embedding.from_pretrained(actor_matrix, freeze=True, max_norm=1.0)
        # Dropout layers
        self.hidden_dropout = nn.Dropout(self.hidden_dropout_p)
        # Word encoder
        self.word_embedding = WordEncoder(fast_text_matrix, 
                                          char_vocab_size=self.char_vocab_size,
                                          char_embedding_dim=self.char_embedding_dim,
                                          char_pad_idx=self.char_pad_index,
                                          filters_per_channel=self.filters_per_channel,
                                          kernels=self.kernels,
                                          lstm_hidden_dim=self.lstm_hidden_dim, 
                                          num_lstm_layers=self.num_lstm_layers, 
                                          embedding_dropout_p=self.embedding_dropout_p,
                                          hidden_dropout_p=self.hidden_dropout_p)
        #
        self.entity_feature_embedding = nn.Embedding(num_embeddings=self.num_efeatures,
                                                     embedding_dim=self.efeature_embedding_dim,
                                                     max_norm=1.0)
        self.claim_feature_embedding = nn.Embedding(num_embeddings=self.num_cfeatures,
                                                    embedding_dim=self.cfeature_embedding_dim,
                                                    max_norm=1.0)
        self.distance_feature_embedding = nn.Embedding(num_embeddings=self.num_dfeatures,
                                                       embedding_dim=self.dfeature_embedding_dim,
                                                       max_norm=1.0)
        nn.init.xavier_uniform_(self.entity_feature_embedding.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.claim_feature_embedding.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.distance_feature_embedding.weight.data, gain=math.sqrt(2))
        #
        self.alpha_encoder = SingleQueryEncoder(dimension=self.word_embedding_dim,
                                                hidden_size=self.ffnn_hidden_dim,
                                                num_layers=self.num_ffnn_layers,
                                                dropout_p=self.hidden_dropout_p)
        self.gamma_encoder = MultiQueryEncoder(dimension=self.word_embedding_dim,
                                               num_queries=self.num_cfeatures,
                                               hidden_size=self.ffnn_hidden_dim,
                                               num_layers=self.num_ffnn_layers,
                                               dropout_p=self.hidden_dropout_p)
        # Sub-components for score_link
        span_encoding_dim = self.word_embedding_dim*3+self.cfeature_embedding_dim
        self.affine_ent = nn.Linear(in_features=span_encoding_dim,
                                    out_features=self.word_embedding_dim)
        #
        
        # Sub-components for score_rel
        rel_dim = 3*span_encoding_dim + self.dfeature_embedding_dim
        self.ffnn_rel = FFNN(input_size=rel_dim,
                             hidden_size=self.ffnn_hidden_dim,
                             num_layers=self.num_ffnn_layers,
                             dropout_p=self.hidden_dropout_p,
                             projection_dim=1)
        # 
        self.context_encoder = None
        self.average_link = None
        if self.use_context is True:
            self.context_encoder = AffineMultiQueryHardAttentionEncoder(dimension=self.word_embedding_dim,
                                                                        k=self.K_context,)
            self.average_link = self.average = nn.Linear(in_features=2,
                                                         out_features=1)
    
    def _zeros(self, dim, cuda_flag):
        if cuda_flag is True:
            return torch.zeros(dim).cuda()
        return torch.zeros(dim)
    
    def to(self, device):
        self.word_embedding.to(device)
        self.actor_embedding.to(device)
        self.entity_feature_embedding.to(device)
        self.claim_feature_embedding.to(device)
        self.distance_feature_embedding.to(device)
        self.alpha_encoder.to(device)
        self.gamma_encoder.to(device)
        self.affine_ent.to(device)
        self.ffnn_rel.to(device)
        if self.use_context is True:
            self.context_encoder.to(device)
            self.average_link.to(device)
        return self
    
    def context_encodings(self, espans, actor_embeddings, word_vectors):
        encodings = []
        for span in espans:
            s = max(0, span[0]-self.context_size)
            e = min(word_vectors.shape[0], span[1]+self.context_size)
            left_context  = word_vectors[s:span[0]]
            right_context = word_vectors[span[1]:e]
            context = torch.cat([left_context, right_context], dim=0)
            context_encoding, _ = self.context_encoder(actor_embeddings, context, context)
            encodings.append(context_encoding.view(1,-1))
        encodings = torch.cat(encodings, dim=0)
        return encodings
    
    def espan_encodings(self, espans, keys, values):
        encodings = []
        for span in espans:
            xf = keys[span[0]].view(1,-1)
            xl = keys[span[1]-1].view(1,-1)
            fs = self.entity_feature_embedding(span[2]).view(1,-1)
            fs = self.hidden_dropout(fs)
            xh = self.alpha_encoder(keys[span[0]:span[1]], values[span[0]:span[1]]).view(1,-1)
            encodings.append(torch.cat([xf,xl,xh,fs], dim=1))
        encodings = torch.cat(encodings, dim=0)
        return encodings
    
    def cspan_encodings(self, cspan, espans, word_vectors, value_vectors):
        # first and last vector of of span
        xf = word_vectors[cspan[0]].view(1,-1)
        xl = word_vectors[cspan[1]-1].view(1,-1)
        # feature vector over all claim categories
        feats = self.claim_feature_embedding(cspan[2])
        feats = self.hidden_dropout(feats)
        
        # soft head for each claim category
        xhs = self.gamma_encoder(query_ids=cspan[2],
                                 keys=word_vectors[cspan[0]:cspan[1]],
                                 values=value_vectors[cspan[0]:cspan[1]])
        # concate all together to create a claim-encoding for each claim-category
        claim_encodings = []
        for i in range(xhs.shape[0]):
            xh = xhs[i].view(1,-1)
            fs = feats[i].view(1,-1)
            claim_encodings.append(torch.cat([xf,xl,xh,fs], dim=1))
        claim_encodings = torch.cat(claim_encodings, dim=0)
        #
        return claim_encodings
    
    def link_scores(self, espans, entity_span_encodings, actor_ids, word_embeddings):
        small_entity_encodings = self.affine_ent(entity_span_encodings)
        # Select embeddings of attributed actors
        actor_embeddings = self.actor_embedding(actor_ids)
        actor_embeddings = self.hidden_dropout(actor_embeddings)
        # Calculate score_link for all actors in doc
        score_link = self.dot_product(actor_embeddings, small_entity_encodings)
        # Calculate score link on basis of context attention
        if self.use_context:
            # Create encodings for contexts of entity spans
            entity_context_encodings = self.context_encodings(espans, actor_embeddings, word_embeddings)
            # Calculate link score based on entity context encodings
            context_scores = self.dot_product(actor_embeddings, entity_context_encodings)
            # Merge link scores
            score_link = torch.cat([score_link.unsqueeze(-1), context_scores.unsqueeze(-1)], dim=2)
            score_link = self.average(score_link).squeeze(-1)
        return score_link
    
    def relation_scores(self, espan_encodings, cspan_encodings, dfeatures):
        num_espans = espan_encodings.shape[0]
        num_ccats = cspan_encodings.shape[0]
        rel_scores = []
        inputs = []
        #
        for i in range(num_espans):
            dfeat = self.distance_feature_embedding(dfeatures[i]).view(1, -1)
            dfeat = self.hidden_dropout(dfeat)
            e = espan_encodings[i].view(1, -1)
            for j in range(num_ccats):
                c = cspan_encodings[j].view(1, -1)
                s = torch.cat([e, c, e*c, dfeat], dim=1)
                inputs.append(s)
        inputs = torch.cat(inputs, dim=0)
        rel_scores = self.ffnn_rel(inputs).view(num_espans, num_ccats)
        return rel_scores
    
    def forward(self, word_ids, char_ids, attributed_actors, espans, cspans, dfeatures):
        # Calculate contextualised word-embeddings
        word_embeddings, fsttxt_embeddings = self.word_embedding(word_ids, char_ids)
        # Calculate entity-span-encodings
        entity_span_encodings = self.espan_encodings(espans, keys=word_embeddings, values=fsttxt_embeddings)
        # Calculate score_rel for all entities and claims in doc
        score_rs = []
        # For each given claim
        for i in range(len(cspans)):
            # Input data for claim i
            actor_ids = attributed_actors[i]
            claim_span = cspans[i]
            dfeats = dfeatures[:,i]
            # If not actor is attibuted to this claim
            if actor_ids.shape[0] == 0:
                # then skip it
                score_rel = torch.empty(0, claim_span[2].shape[0])
                tmp = torch.empty(0, len(espans), claim_span[2].shape[0])
                # by outputing empty tensors
                score_rs.append((score_rel, tmp))
                continue
            # Calculate score_link for each attributed actor
            score_link = self.link_scores(espans, entity_span_encodings, actor_ids, word_embeddings)
            # Calculate a claim-embedding for each claim-categorie annotated to this claim
            claim_span_encodings = self.cspan_encodings(claim_span, espans, word_vectors=word_embeddings, value_vectors=fsttxt_embeddings)
            # Calculate score_rel for each actor and each categorie of this laim
            score_rel =  self.relation_scores(entity_span_encodings, claim_span_encodings, dfeats)
            # Combine score_link and score_rel
            tmp = score_link.unsqueeze(-1) + score_rel.unsqueeze(0)
            # Aggregate over actors
            score4 = self.aggregate(tmp, dim=1)
            score_rs.append((score4, tmp))
        # Return list of score_4 for each claim
        return score_rs
