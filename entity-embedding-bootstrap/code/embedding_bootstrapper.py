"""
This file contains the BootstrapModel class. This class is used to bootstrap the entity-embeddings from fasttext-embeddings.
"""
import torch
import torch.optim as optim

import numpy as np

import math
import json
import re
import os

from .counter import Counter
from .bootstrap_model import BootstrapModel

import matplotlib.pyplot as plt


class EmbeddingBootstrapper(object):
    """
    This class is used to bootstrap entity-embeddings from fasttext embeddings.
    It provides methods for loading the required data and train the entity-embeddings.
    """
    def __init__(self, fasttext_embeddings, fasttext_wmap, word_counts, entity_embeddings=None, entity_map=None,
                 smoothing_exp=0.75, margin=0.1, num_pos_sampels=20, num_neg_sampels=5, 
                 one_norm=False, sec_param=None, max_norm=None):
        """
        """
        # Embedding-Data
        self.fasttext_wmap = fasttext_wmap
        self.fasttext_embeddings = fasttext_embeddings
        self.entity_embeddings = entity_embeddings
        self.entity_map = entity_map
        # Hyper-Paramters
        self.smoothing_exp = smoothing_exp
        self.margin = margin
        self.num_pos_sampels = num_pos_sampels
        self.num_neg_sampels = num_neg_sampels
        self.one_norm = one_norm
        self.sec_param = sec_param
        self.max_norm = max_norm
        # Unigram word distribution
        self.word_dist = self._sum2one_(word_counts, dim=0)   # make a distribution from counts
        self.word_dist = self._sum2one_(np.power(self.word_dist, self.smoothing_exp), dim=0)   # smooth the distribution
    
    def train(self, data, num_epochs, learning_rate=1.0, stop_iterations=None, stop_threshold=1e-5):
        """
        Trains entity-embeddings on the given data. For training a NoiseContrastiveLoss is used. The probabilities are estimated from the given data.
        If cuda-functionality is available, it is used.
        """
        # Prepare pytorch cuda
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            print(" > WARNING: CUDA is not available. CPU is used instead.")
        #### Step 1 - Create word-distributions
        # Count wort-entity-cooccurence
        coocc, e_map, w_map = self._measure_cooccurrence_(data, entity_map=self.entity_map)
        debug_map=dict([(i,w) for w,i in w_map.items()])
        # Estimate distributions from measurements
        word_entity_dist = self._sum2one_(coocc, dim=1)
        assert(word_entity_dist.shape[1]==self.word_dist.shape[0])
        #### Step 2 - Train entity-embeddings via hinge-loss
        # Create pytroch-BootstrapModel (triple rankin loss with buildin word-sampler)
        bootstrap = BootstrapModel(word_entity_dist, self.word_dist, self.fasttext_embeddings, self.entity_embeddings,
                                   margin=self.margin, num_pos_sampels=self.num_pos_sampels, num_neg_sampels=self.num_neg_sampels,
                                   sec_param=self.sec_param, max_norm=self.max_norm).to(device)
        # Create pytroch-optimizer
        optimizer = optim.Adagrad(bootstrap.parameters(), lr=learning_rate)
        
        #debug 
        last = torch.ones(300,1)
        
        # Run training-epochs
        if stop_iterations is not None:
            hist = dict([(e, [0,]*stop_iterations) for e in list(data.keys())])
        else:
            hist = dict([(e, None) for e in list(data.keys())])
        entities_to_process = list(hist.keys())
        for i in range(len(entities_to_process)):
            e = entities_to_process[i]
            print(" > Training for Entity ({:>3d}/{:<d})   {:<s}".format(i, len(entities_to_process), e))
            #
            debug1 = []
            debug2 = []
            #
            for epoch in range(num_epochs):
                i = e_map[e]
                # Prepare input (create tensor with entity-idx)
                input_idx = torch.LongTensor([i,]).to(device)
                # Prepare gradients
                optimizer.zero_grad()
                # Calculate expected hinge-loss over sampeled words
                l = bootstrap(input_idx)
                # If no words to sampel are available (site was not liked any where)
                if l.requires_grad is False:
                    continue
                # Calculate gradient
                l.backward()
                # Update parameters
                optimizer.step()
                if self.one_norm is True:
                    bootstrap._normalize_()
                l = l.item()
                # Schedule lr if needed
                debug1.append(l)
                # Make informative print about current loss
                print("\tStep {:>4d}   Loss {:.6f}".format(epoch, l), end="   ")
                # Check knns of anchor
                k=5
                with torch.no_grad():
                    ent_emb = bootstrap.entity_embeddings(input_idx).view(-1, 1).cpu()
                    euclid = torch.linalg.norm(ent_emb-last)
                    cos = 1-((torch.dot(ent_emb.view(-1), last.view(-1)))/(torch.linalg.norm(ent_emb.view(-1))*torch.linalg.norm(last.view(-1))))
                    debug2.append(cos)
                    last = ent_emb
                    sims = torch.matmul(self.fasttext_embeddings, ent_emb).view(-1)
                    idxs = torch.argsort(sims)
                    idxs = idxs[-k-1:-1]
                    knns = [debug_map[i.item()] for i in idxs]
                    print("Euclid {:.6f}   Cos {:.6f}   KNNs {}".format(euclid, cos, knns))
                # Check for convergence of cos-dist-changes
                if stop_iterations is not None:
                    best = sum(hist[e])/len(hist[e])
                    flags = [abs(cos - d)<=stop_threshold for d in hist[e]]
                    if all(flags):
                        # if convergence
                        print("\tConvergence !")
                        # stop training of this entity
                        del hist[e]
                        break
                    else:
                        hist[e] = hist[e][1:]+[cos,]
            # DEBUG
            #plt.subplot(121)
            #plt.title("Loss")
            #plt.plot(np.arange(len(debug1)), debug1)
            #plt.subplot(122)
            #plt.title("Changes")
            #plt.plot(np.arange(len(debug2)), debug2)
            #plt.show()
            print()
            # stop training if all entities stoped training
            if (stop_iterations is not None) and (len(hist)==0):
                break
        
        #### Step 3 - Return the trained embeddings
        self.entity_map = e_map
        self.entity_embeddings = bootstrap.entity_embeddings.weight.detach()
        return self.entity_embeddings
    
    def store_embeddings(self, out_path):
        """
        This method can be called after training. Stores the trained embeddings to the specified file.
        The vectors are stored in a text-format with one vector per line.
        """
        with open(out_path, "tw") as ofile:
            for k,v in self.entity_map.items():
                vec = self.entity_embeddings[v].tolist()
                line = "{} {}".format(k,
                                      " ".join([str(e) for e in vec]))
                ofile.write(line+"\n")
        return None
    
    @staticmethod
    def load_fasttext(path, size=300, renorm=None):
        """
        Load fasttext-embeddings from specified file.
        Returns a tensor holding the vectors and a dictionary containing the word-indices.
        If renorm is not None, the embeddings are normalized such that the magnitude of each vector is renorm.
        """
        wmap = dict()
        data = []
        with torch.no_grad():  # not sure if needed
            with open(path, 'tr', encoding='utf-8') as ifile:
                #
                for line in ifile:
                    #
                    values = line.rstrip().split(' ')
                    #
                    if len(values) < size+1:
                        continue
                    #
                    word = values[0]
                    if word not in wmap:
                        wmap[word] = len(wmap)
                    #
                    values = values[1:]
                    vector = torch.tensor([[float(v) for v in values],])
                    data.append(vector)
            data = torch.cat(data, dim=0)
            if renorm is not None:
                norms = torch.linalg.vector_norm(data, dim=1, keepdims=True)
                data /=  norms
                data *= renorm
        return data, wmap
    
    @staticmethod
    def load_wiki_info_data(path):
        """
        Loads, tokenizes and normalizes the wikipedia-information-pages.
        The specified path must point to a directory which holds data generated with the retrieve_wiki_data.py script.
        For tokenization and normalization spacy is used.
        Normalization includes: Removing stopwords, punctuation and space-characters.
        """
        #
        data_dict = {}
        for file in os.listdir(path):
            file_path = os.path.join(path,file)
            if "." in file_path:
                if file_path.split(".")[-1] == "json":
                    entity_data = EmbeddingBootstrapper._load_entity_file_(file_path)
                    data_dict[entity_data["entity_id"]] = entity_data["description_tokens"]
        return data_dict
    
    @staticmethod
    def load_wiki_link_data(path, include_mention=False):
        """
        Loads, tokenizes and normalizes the wikipedia-link-data.
        The specified path must point to a directory which holds data generated with the retrieve_wiki_data.py script.
        For tokenization and normalization spacy is used.
        Normalization includes: Removing stopwords, punctuation and space-characters.
        """
        # load spacy
        data_dict = {}
        empty_entities = []
        for file in os.listdir(path):
            file_path = os.path.join(path,file)
            if "." in file_path:
                if file_path.split(".")[-1] == "json":
                    entity_data = EmbeddingBootstrapper._load_entity_file_(file_path)
                    # merge context-window-bags
                    tokens = []
                    for link_data in entity_data["link_data"]:
                        tokens += link_data["left_context"]
                        if include_mention is True:
                            tokens += link_data["mention_text"]
                        tokens += link_data["right_context"]
                    data_dict[entity_data["entity_id"]] = tokens
                    if len(tokens) == 0:
                        empty_entities.append(entity_data["entity_id"])
        #
        if len(empty_entities) > 0:
            print(" WARNING: Entities with no link data are ignored. {}".format(empty_entities))
            for e in empty_entities:
                del data_dict[e]
        return data_dict
    
    @staticmethod
    def _load_entity_file_(path):
        """
        Helper method. Loads a single entity-wikidata-file.
        """
        with open(path, "tr", encoding="utf8") as ifile:
            for line in ifile:
                data = json.loads(line.strip())
                return data
        return None
    
    def _measure_cooccurrence_(self, data_dict, entity_map=None):
        """
        Helper function.
        Counts the cooccurence of entities and words in the given data.
        Only words that are in the fasttext-lexicon are counted.
        """
        counter = Counter(word_map=self.fasttext_wmap, entity_map=entity_map)
        # for each entity count words in given text
        for entity, tokens in data_dict.items():
            # count words
            for t in tokens:
                if t in self.fasttext_wmap:  # only if they are in the vocab
                    counter(entity, t)
        # return counts (numpy-array), entity_map (dict), word_map (dict)
        return counter.to_numpy()
    
    def load_words_counts(path, word_map):
        counts = np.zeros((len(word_map),))
        with open(path, "tr") as ifile:
            for line in ifile:
                t,c = line.rstrip("\n").split(" ")
                c = int(c)
                if t in word_map:
                    counts[word_map[t]] = c
        return counts
    
    @staticmethod
    def _sum2one_(a, dim=0):
        b = np.sum(a, axis=dim, keepdims=True)
        mask = b>0
        c = np.divide(a, b, out=np.zeros_like(a), where=mask)
        return c

