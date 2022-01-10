#!/usr/bin/python3
"""
This file holds a class for the first model.

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


class Model1(nn.Module):
    """
    This class provides the first model form the bachelor thesis (section 4.7)
    """
    def __init__(self, fast_text_matrix, actor_matrix, char_vocab_size, num_efeatures, char_pad_idx,
                       efeature_embedding_dim,
                       char_embedding_dim, filters_per_channel, kernels,
                       num_lstm_layers, ffnn_hidden_dim, num_ffnn_layers,
                       embedding_dropout_p, hidden_dropout_p,
                       use_context, context_size, K):
        super(Model1, self).__init__()
        # Hyper-Parameters (Input)
        self.word_vocab_size = fast_text_matrix.shape[0]
        self.word_embedding_dim = fast_text_matrix.shape[1]
        self.num_efeatures = num_efeatures   # Number of entity-feature-embeddings
        self.char_vocab_size = char_vocab_size
        self.char_pad_idx = char_pad_idx
        #
        self.use_context = use_context
        self.context_size = context_size
        self.K = K
        # Dimension of entity-feature-embeddings
        self.efeature_embedding_dim = efeature_embedding_dim  
        # Hyper-Parameters for Word-Embeddings
        self.char_embedding_dim = char_embedding_dim
        self.filters_per_channel = filters_per_channel
        self.kernels = kernels
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dim = self.word_embedding_dim/2
        # Hyper-Parameters for all FFNNs
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.num_ffnn_layers = num_ffnn_layers 
        # Method used to calculate dot-product (in this case normal dot-product)
        self.dot_product = DotProduct.simpel
        # Method used to aggregate over scores ('lse' or 'max')
        self.aggregate = Maximum.lse
        # Dropout applied to word- and char-embeddings
        self.embedding_dropout_p = embedding_dropout_p
        # Dropout applied hidden layers and feature embeddings
        self.hidden_dropout_p = hidden_dropout_p
        # Pretrained-Embeddings (Input)
        self.num_actors = actor_matrix.shape[0]
        self.actor_embedding = nn.Embedding.from_pretrained(actor_matrix, freeze=True, max_norm=1.0)
        # Dropout layers
        self.feature_dropout = nn.Dropout(self.hidden_dropout_p)
        self.actor_dropout = nn.Dropout(self.hidden_dropout_p)
        # Word encoder
        self.word_embedding = WordEncoder(fast_text_matrix, 
                                          char_vocab_size=self.char_vocab_size,
                                          char_pad_idx=self.char_pad_idx,
                                          char_embedding_dim=self.char_embedding_dim, 
                                          filters_per_channel=self.filters_per_channel,
                                          kernels=self.kernels,
                                          lstm_hidden_dim=self.lstm_hidden_dim, 
                                          num_lstm_layers=self.num_lstm_layers, 
                                          embedding_dropout_p=self.embedding_dropout_p,
                                          hidden_dropout_p=self.hidden_dropout_p)
        # Sub-components for span-encodings
        self.entity_feature_embedding = nn.Embedding(num_embeddings=self.num_efeatures,
                                                     embedding_dim=self.efeature_embedding_dim,
                                                     max_norm=1.0)
        nn.init.xavier_uniform_(self.entity_feature_embedding.weight.data, gain=math.sqrt(2))
        self.alpha_encoder = SingleQueryEncoder(dimension=self.word_embedding_dim,
                                                hidden_size=self.ffnn_hidden_dim,
                                                num_layers=self.num_ffnn_layers,
                                                dropout_p=self.hidden_dropout_p)
        # Sub-components for score_link
        span_encoding_dim = self.word_embedding_dim*3+self.efeature_embedding_dim*2 
        self.affine_ent = nn.Linear(in_features=span_encoding_dim,
                                    out_features=self.word_embedding_dim,
                                    bias=True)
        # Sub-components for context attention
        if self.use_context:
            self.context_encoder = AffineMultiQueryHardAttentionEncoder(dimension=self.word_embedding_dim,
                                                                        k=self.K,)
            self.average = nn.Linear(in_features=2,
                                     out_features=1)
        else:
            self.context_encoder = None
            self.average = None
    
    def to(self, device):
        self.word_embedding.to(device)
        self.actor_embedding.to(device)
        self.entity_feature_embedding.to(device)
        self.alpha_encoder.to(device)
        #
        self.affine_ent.to(device)
        #
        if self.use_context is True:
            self.average.to(device)
            self.context_encoder.to(device)
        return self
    
    def espan_encodings(self, espans, keys, values):
        encodings = []
        for span in espans:
            xf = keys[span[0]].view(1,-1)
            xl = keys[span[1]-1].view(1,-1)
            fs = self.entity_feature_embedding(span[2]).view(1,-1)
            fs = self.feature_dropout(fs)
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
            context_encoding, _ = self.context_encoder(actor_embeddings, context, context)  # ques can be retrieved here
            encodings.append(context_encoding.view(1,-1))
        encodings = torch.cat(encodings, dim=0)
        return encodings
    
    def all_actor_embeddings(self, cuda_flag):
        all_actor_ids = torch.arange(self.num_actors)
        if cuda_flag is True:
            all_actor_ids = all_actor_ids.cuda()
        actor_embeddings = self.actor_embedding(all_actor_ids)
        actor_embeddings = self.actor_dropout(actor_embeddings.unsqueeze(0)).squeeze(0)
        return actor_embeddings
    
    def forward(self, word_ids, char_ids, espans):
        # Create word embeddings
        word_embeddings, fsttxt_embeddings = self.word_embedding(word_ids, char_ids)
        # Load actor embeddings
        actor_embeddings = self.all_actor_embeddings(word_ids.is_cuda)
        # Create encodings for entity spans
        entity_span_encodings = self.espan_encodings(espans, keys=word_embeddings, values=fsttxt_embeddings)
        # Transform span-encodings to same size as actor-embeddings
        small_entity_encodings = self.affine_ent(entity_span_encodings)
        # Calculate link score based on entity encodings
        score_link = self.dot_product(actor_embeddings, small_entity_encodings)
        #
        if self.use_context:
            # Create encodings for contexts of entity spans
            entity_context_encodings = self.espan_context_encodings(espans, actor_embeddings, word_embeddings)
            # Calculate link score based on entity context encodings
            context_scores = self.dot_product(actor_embeddings, entity_context_encodings)
            # Merge link scores
            score_link = torch.cat([score_link.unsqueeze(-1), context_scores.unsqueeze(-1)], dim=2)
            score_link = self.average(score_link).squeeze(-1)
        #
        score_1 = self.aggregate(score_link, dim=1)
        #
        return score_1, (score_link,)  # return score_link as only hidden state

