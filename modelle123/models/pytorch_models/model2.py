#!/usr/bin/python3
"""
This file holds a class for the second model.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import math
import torch
import torch.nn as nn

from .ffnn import FFNN
from .word_encoder import WordEncoder
from .single_query_encoder import SingleQueryEncoder
from .dot_product import DotProduct
from .maximum import Maximum
from .hard_attention import AffineMultiQueryHardAttentionEncoder



class Model2(nn.Module):
    """
    This class provides the second model form the bachelor thesis (section 4.8)
    """
    def __init__(self, fast_text_matrix, actor_matrix, char_vocab_size, char_pad_index, num_efeatures, num_cfeatures, num_dfeatures,
                       efeature_embedding_dim, cfeature_embedding_dim, dfeature_embedding_dim,
                       char_embedding_dim, filters_per_channel, kernels, num_lstm_layers,
                       ffnn_hidden_dim, num_ffnn_layers,
                       embedding_dropout_p, hidden_dropout_p,
                       use_context, context_size, K_context):
        super(Model2, self).__init__()
        # Hyper-Parameters (Input)
        self.word_vocab_size = fast_text_matrix.shape[0]
        self.word_embedding_dim = fast_text_matrix.shape[1]
        
        self.num_efeatures = num_efeatures   # Number of entity-feature-embeddings
        self.num_cfeatures = num_cfeatures   # Number of claim-feature-embeddings
        self.num_dfeatures = num_dfeatures   # Number of distance-feature-embeddings
        self.char_vocab_size = char_vocab_size
        self.char_pad_index = char_pad_index
        
        self.efeature_embedding_dim = efeature_embedding_dim  # Dimension of entity-feature-embeddings
        self.cfeature_embedding_dim = cfeature_embedding_dim  # Dimension of claim-feature-embeddings
        self.dfeature_embedding_dim = dfeature_embedding_dim  # Dimension of distance-feature-embeddings
        
        self.char_embedding_dim = char_embedding_dim
        self.filters_per_channel = filters_per_channel
        self.kernels = kernels
        self.num_lstm_layers = num_lstm_layers 
        self.lstm_hidden_dim = self.word_embedding_dim/2
        
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.num_ffnn_layers = num_ffnn_layers
        
        self.dot_product = DotProduct.simpel
        self.aggregate = Maximum.lse
        
        self.use_context = use_context
        self.context_size = context_size
        self.K_context = K_context
        
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
        self.beta_encoder = SingleQueryEncoder(dimension=self.word_embedding_dim,
                                               hidden_size=self.ffnn_hidden_dim,
                                               num_layers=self.num_ffnn_layers,
                                               dropout_p=self.hidden_dropout_p)
        # Sub-components for score_link
        span_encoding_dim = self.word_embedding_dim*3+self.efeature_embedding_dim*2
        self.affine_ent = nn.Linear(in_features=span_encoding_dim,
                                    out_features=self.word_embedding_dim)
        #
        self.context_encoder = None
        self.average_link = None
        if self.use_context is True:
            self.context_encoder = AffineMultiQueryHardAttentionEncoder(dimension=self.word_embedding_dim,
                                                                        k=self.K_context,)
            self.average_link = self.average = nn.Linear(in_features=2,
                                                         out_features=1)
        # Sub-components for score_attr
        attr_size = 3*span_encoding_dim+self.dfeature_embedding_dim
        self.ffnn_attr = FFNN(input_size=attr_size,
                              hidden_size=self.ffnn_hidden_dim,
                              num_layers=self.num_ffnn_layers,
                              dropout_p=self.hidden_dropout_p,
                              projection_dim=1)
    
    def _zeros(self, dim, cuda_flag):
        if cuda_flag is True:
            return torch.zeros(dim).cuda()
        return torch.zeros(dim)
    
    def to(self, device):
        self.word_embedding.to(device)
        self.actor_embedding.to(device)
        self.entity_feature_embedding.to(device)
        self.claim_feature_embedding.to(device)
        self.alpha_encoder.to(device)
        self.beta_encoder.to(device)
        self.affine_ent.to(device)
        self.distance_feature_embedding.to(device)
        self.ffnn_attr.to(device)
        if self.use_context is True:
            self.context_encoder.to(device)
            self.average_link.to(device)
        return self
    
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
    
    def espan_context_encodings(self, espans, actor_embeddings, word_vectors):
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
    
    def cspan_encodings(self, cspans, keys, values):
        encodings = []
        for span in cspans:
            xf = keys[span[0]].view(1,-1)
            xl = keys[span[1]-1].view(1,-1)
            fs = self.claim_feature_embedding(span[2])
            fs = self.hidden_dropout(fs)
            fs = torch.sum(fs, dim=0).view(1,-1)
            xh = self.beta_encoder(keys[span[0]:span[1]], values[span[0]:span[1]]).view(1,-1)
            encodings.append(torch.cat([xf,xl,xh,fs], dim=1))
        encodings = torch.cat(encodings, dim=0)
        return encodings
    
    def attribution_scores(self, espan_encodings, cspan_encodings, dfeatures):
        num_espans = espan_encodings.shape[0]
        num_cspans = cspan_encodings.shape[0]
        # For each entity-claim-pair prepair input to ffnn_attr
        inputs = []
        for i in range(num_espans):
            for j in range(num_cspans):
                e = espan_encodings[i].view(1, -1)
                c = cspan_encodings[j].view(1, -1)
                dfeat = self.distance_feature_embedding(dfeatures[i,j]).view(1, -1)
                dfeat = self.hidden_dropout(dfeat)
                inputs.append(torch.cat([e, c, e*c, dfeat], dim=1))
        inputs = torch.cat(inputs, dim=0)
        # Run inputs through ffnn and reorder them into a matrix
        attribution_scores = self.ffnn_attr(inputs).view(num_espans, num_cspans)
        return attribution_scores
    
    def forward(self, word_ids, char_ids, candidate_ids, espans, cspans, dfeatures, verbose=False, intend=""):
        # Create word embeddings
        word_embeddings, fsttxt_embeddings = self.word_embedding(word_ids, char_ids)
        # Create span-encodings for entities and claims
        entity_span_encodings = self.espan_encodings(espans, keys=word_embeddings, values=fsttxt_embeddings)
        claim_span_encodings = self.cspan_encodings(cspans, keys=word_embeddings, values=fsttxt_embeddings)
        # Calculate the Score_attr
        score_attr = self.attribution_scores(entity_span_encodings, claim_span_encodings, dfeatures)
        # If candidate set is empty
        if candidate_ids.shape[0] == 0:
            # Create score_2
            score_2   = torch.empty(0, len(cspans))
            score_linked_attr = torch.empty(0, len(espans), len(cspans))
            # Create score_nil
            score_nil = torch.full((len(cspans),), float("+inf"))
            # Return score_attr as the hidden_state instead of score_nil_attr
            return score_2, score_nil, (score_linked_attr, score_attr)
        # Else ...
        # Create actor embeddings for candidates
        actor_embeddings = self.actor_embedding(candidate_ids)
        actor_embeddings = self.hidden_dropout(actor_embeddings)
        # Calculate the Score_link
        small_entity_encodings = self.affine_ent(entity_span_encodings)
        score_link = self.dot_product(actor_embeddings, small_entity_encodings)
        # Calculate score link on basis of context attention
        if self.use_context:
            # Create encodings for contexts of entity spans
            entity_context_encodings = self.espan_context_encodings(espans, actor_embeddings, word_embeddings)
            # Calculate link score based on entity context encodings
            context_scores = self.dot_product(actor_embeddings, entity_context_encodings)
            # Merge link scores
            score_link = torch.cat([score_link.unsqueeze(-1), context_scores.unsqueeze(-1)], dim=2)
            score_link = self.average(score_link).squeeze(-1)
        # Calculate the Score_attr_link
        score_linked_attr = score_link.unsqueeze(-1) + score_attr.unsqueeze(0)
        # Else  calculate score_nil_attr
        score_linkable = Maximum.max(score_link, dim=0)
        score_nil_attr = score_attr - score_linkable.unsqueeze(-1)
        # Calculate score_nil
        score_nil = self.aggregate(score_nil_attr, dim=0)
        # Calculate score_2
        score_2 = self.aggregate(score_linked_attr, dim=1)
        #
        return score_2, score_nil, (score_linked_attr, score_nil_attr)  # if actor candidate set IS NOT empty return nil_attr_scores
