#!/usr/bin/python3
"""
This file holds a word embedding class. Words are encoded as described in the bachelor thesis (section 4.6).

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""


import math
import torch
import torch.nn as nn

from .highway_layer import HighwayLayer

class WordEncoder(nn.Module):
    """
    This is a pytorch class for representing words as vectors. For details see section 4.6 of bachelor thesis.
    This class requires fasttext vectors.
    """
    def __init__(self, fast_text_matrix, char_vocab_size, char_pad_idx,
                 char_embedding_dim, filters_per_channel, kernels,
                 lstm_hidden_dim, num_lstm_layers, 
                 embedding_dropout_p, hidden_dropout_p):
        super(WordEncoder, self).__init__()
        #
        self.word_vocab_size = fast_text_matrix.shape[0]
        self.word_embedding_dim = fast_text_matrix.shape[1]
        self.char_vocab_size = int(char_vocab_size)
        self.char_embedding_dim = int(char_embedding_dim)
        self.filters_per_channel = int(filters_per_channel)
        self.num_lstm_layers = int(num_lstm_layers)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.embedding_dropout_p = embedding_dropout_p
        self.hidden_dropout_p = hidden_dropout_p
        self.char_pad_idx = char_pad_idx
        #
        self.fast_text_embedding = nn.Embedding.from_pretrained(fast_text_matrix, freeze=True, max_norm=1.0)
        #
        self.char_embedding = nn.Embedding(num_embeddings=self.char_vocab_size,
                                           embedding_dim=self.char_embedding_dim,
                                           padding_idx=self.char_pad_idx,
                                           max_norm=1.0)
        nn.init.xavier_uniform_(self.char_embedding.weight.data, gain=math.sqrt(2))
        # Convolutional layers
        out_channels = self.char_embedding_dim * self.filters_per_channel
        self.convolutions = []
        for length in kernels:
            conv = nn.Conv1d(in_channels=self.char_embedding_dim,
                             out_channels=out_channels,
                             kernel_size=length,
                             groups=self.char_embedding_dim,
                             padding="same")
            self.convolutions.append(conv)
        self.convolutions = nn.ModuleList(self.convolutions)
        # Highway-Layer
        self.hnn = HighwayLayer(self.word_embedding_dim+len(kernels)*out_channels)
        # LSTM with diffrent dropout-mask per time-step
        self.lstm = nn.LSTM(input_size=self.word_embedding_dim+len(kernels)*out_channels,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.num_lstm_layers,
                            bidirectional=True,
                            dropout=self.hidden_dropout_p,
                            batch_first=True)
        #
        def init_weights(m):
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
        self.lstm.apply(init_weights)
        # Dropout for FastText-Embeddings and CNN-Output
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_p)
        # Dropout for Hidden Layers (of LSTM and HLayer)
        self.hidden_dropout = nn.Dropout(self.hidden_dropout_p)
    
    def to(self, device):
        self.fast_text_embedding.to("cpu")
        self.char_embedding.to(device)
        self.convolutions.to(device)
        self.hnn.to(device)
        self.lstm.to(device)
        self.embedding_dropout.to(device)
        self.hidden_dropout.to(device)
        return self
    
    def forward(self, word_ids, char_ids):
        num_words = char_ids.shape[0]
        cuda_flag = word_ids.is_cuda
        # Get word embeddings
        if cuda_flag:
            word_ids = word_ids.cpu()
        fsttxt_embeddings = self.fast_text_embedding(word_ids)
        if cuda_flag:
            fsttxt_embeddings = fsttxt_embeddings.cuda()
        # Get char embeddings
        char_embeddings = self.char_embedding(char_ids).transpose(1,2)
        char_features = []
        for conv in self.convolutions:
            feats = torch.max(conv(char_embeddings), dim=2).values
            char_features.append(feats)
        char_features = torch.cat(char_features, dim=1)
        # Concatenate word and char-based embeddings
        fsttxt_embeddings = self.embedding_dropout(fsttxt_embeddings)
        char_features = self.embedding_dropout(char_features)
        word_embeddings = torch.cat([fsttxt_embeddings, char_features], dim=1).unsqueeze(0)
        # Feed concatenation of word and char-based embeddings through highway-layer
        word_embeddings = self.hnn(word_embeddings)
        # Contextualize word-embeddings with LSTM
        context_word_embeddings,_ = self.lstm(word_embeddings)
        context_word_embeddings = context_word_embeddings.squeeze(0)
        context_word_embeddings = self.hidden_dropout(context_word_embeddings)
        #
        return context_word_embeddings, fsttxt_embeddings
