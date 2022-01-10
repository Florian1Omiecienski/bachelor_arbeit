"""
This file contains the BootstrapModel class. This class is a pytorch module and is used for training the entity embeddings.
TODO adujst comments cause init method changed
"""

import torch
import torch.nn as nn

import numpy as np


class BootstrapModel(nn.Module):
    """
    This Class is used for sampling positive and negative words from a distribution and calculating a triplet margin loss for an entity embedding and the sampel words.
    """
    def __init__(self, word_entity_dist, word_dist, fasttext_embeddings, entity_embeddings, margin, num_pos_sampels, num_neg_sampels, sec_param=None, max_norm=None):
        """
        Input:
            word_entity_dist: numpy-array of shape (num_entities, num_words) holding the positive distribution P(w|e)
            word_dist: numpy-array of shape (num_words,) holding the negative distribution P(w)
            fasttext_embeddings: tensor holding the fasttext vectors
            entity_embeddings: tensor holding pretrained entity-embeddings or None
            margin: float, the margin that should separate the distance of entity-emb. to pos-word and the distance of entity-emb. to neg-word
            num_pos_sampels: integer, the number of positive words to sampel in a forward call
            num_neg_sampels: integer, the number of negative words to sampel for each positive sampel
            reg_method: the method to keep the learned embeddings on unit-sphere. allowed values ('None','max_norm')
        """
        super(BootstrapModel, self).__init__()
        #
        # Hyper-Parameters (Tunabel)
        self.num_pos_sampels = num_pos_sampels
        self.num_neg_sampels = num_neg_sampels
        self.margin = margin
        self.sec_param = sec_param
        self.max_norm = max_norm
        # Hyper-Parameters (Fixed by inserted FastText-Embeddings and Word-Counts)
        self.num_entities = word_entity_dist.shape[0]
        self.num_words = word_entity_dist.shape[1]
        self.embedding_size = fasttext_embeddings.shape[1]
        # word-distributions (Used for sampeling words)  (Must be numpy-arrays)
        self.word_entity_dist = word_entity_dist 
        self.word_dist = word_dist
        # FastText-Embedding-Component (Fixed)
        self.fasttext_embeddings = nn.Embedding.from_pretrained(fasttext_embeddings, freeze=True)
        # New or specified Entity-Embeddings (Not fixed)
        print(" DEBUG: max_norm=", self.max_norm)
        if entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(num_embeddings=self.num_entities,
                                                  embedding_dim=self.embedding_size,
                                                  max_norm=self.max_norm)
            nn.init.normal_(self.entity_embeddings.weight, mean=0, std=1)
        else:
            self.entity_embeddings = nn.Embedding.from_pretrained(entity_embeddings, freeze=False, max_norm=self.max_norm)
        print(" DEBUG: Created BootstrapModel with {} embeddings (Finetune={})".format(self.entity_embeddings.weight.shape[0],
                                                                                       entity_embeddings is not None))
    def _normalize_(self):
        """
        Replaces the entity embeddings to a length of one.
        """
        with torch.no_grad():
            new_weight = self.entity_embeddings.weight
            new_weight /= torch.linalg.norm(new_weight, dim=1).view(-1,1)
            self.entity_embeddings.weight = new_weight
    
    def _sampel_positive_(self, entity_idx, cuda):
        """
        Sampels positive word indices for specified entity index
        Returns sampled indices and probs of indices.
        """
        with torch.no_grad():
            pos_idxs = np.random.choice(self.num_words, size=self.num_pos_sampels, p=self.word_entity_dist[entity_idx])
            pos_probs = torch.FloatTensor(self.word_entity_dist[entity_idx, pos_idxs])
            pos_idxs = torch.LongTensor(pos_idxs)
            if cuda:
                pos_probs = pos_probs.cuda()
        return pos_idxs, pos_probs
    
    def _sampel_negative_(self, cuda):
        """
        Sampels negative word indices.
        Returns sampled indices and probs of indices
        """
        with torch.no_grad():
            neg_idxs = np.random.choice(self.num_words, size=self.num_neg_sampels, p=self.word_dist)
            neg_probs = torch.FloatTensor(self.word_dist[neg_idxs].tolist())
            neg_idxs = torch.LongTensor(neg_idxs)
            if cuda:
                neg_probs = neg_probs.cuda()
        return neg_idxs, neg_probs
    
    def _zero_(self, cuda):
        """
        creates a zero tensor
        """
        if cuda:
            return torch.zeros(1).cuda()
        return torch.zeros(1)
    
    def _cos_dist_(self, x1, x2, eps=1e-8):
        """
        Description:
            Calculates the cosine distance between vectors x1 and x2.
            Is trackable wrt. gradients.
            returns tensor of shape (1,)
        """
        n1 = torch.linalg.norm(x1)
        n2 = torch.linalg.norm(x2)
        den = (n1*n2) if (n1*n2)>0 else eps
        cossim = torch.dot(x1, x2)/den
        cosdist = 1-cossim
        return cosdist
    
    def _euc_dist_(self, x1, x2):
        dist = torch.linalg.norm(x2-x1)
        return dist
    
    def forward(self, entity_idx):
        """
        Description:
            For the specified entity_idx some positive and some negative words are sampled.
            For these sampels a triplet ranking loss is calculated and returend.
        Input:
            entity_idx: A LongTensor of shape (1,) that holds a entity index
        Returns:
            A FloatTensor of shape (1,) holding the calculated loss.
        """
        cuda_flag = entity_idx.is_cuda
        # get entity_vector
        evec = self.entity_embeddings(entity_idx).squeeze()
        # sample positive words
        pos_word_idxs, pos_probs = self._sampel_positive_(entity_idx, cuda_flag)
        #print("Pos:",[self.debug_map[pi.item()] for pi in pos_word_idxs])
        pos_vecs = self.fasttext_embeddings(pos_word_idxs)
        if cuda_flag:
            pos_vecs = pos_vecs.cuda()
        # calculate expected loss over all positive words
        loss = self._zero_(cuda_flag)
        #
        if self.sec_param is not None:
            all_norms = [torch.linalg.norm(evec).view(1), torch.linalg.norm(pos_vecs, dim=1).view(-1)]
        #
        for j in range(self.num_pos_sampels):
            # sampel negative words
            neg_word_idxs, neg_probs = self._sampel_negative_(cuda_flag)
            neg_vecs = self.fasttext_embeddings(neg_word_idxs)
            if cuda_flag:
                neg_vecs = neg_vecs.cuda()
            # 
            if self.sec_param is not None:
                all_norms.append(torch.linalg.norm(neg_vecs, dim=1).view(-1))
            # calculate expected loss over all negative words
            negative_loss = self._zero_(cuda_flag)
            for k in range(self.num_neg_sampels):
                pvec = pos_vecs[j]
                nvec = neg_vecs[k]
                # calculate similarity-distance
                pdist = self._cos_dist_(evec, pvec) ########
                ndist = self._cos_dist_(evec, nvec) ########
                # calculate hinge-loss of pos-neg-pair
                l = torch.relu(self.margin + pdist - ndist)
                negative_loss = negative_loss + (neg_probs[k]*l)
            loss = loss + (pos_probs[j]*negative_loss)
        #
        if self.sec_param is not None:
            mean_norm = torch.mean(torch.cat(all_norms))
            ent_norm = torch.linalg.norm(evec)
            spherical_constrain = torch.square(ent_norm-mean_norm)
            loss = loss + self.sec_param*spherical_constrain
        return loss
    
    def to(self, device):
        """
        Sets the module on specified device.
        """
        self.entity_embeddings.to(device)
        return self
