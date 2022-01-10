#!/usr/bin/python3
"""
This file holds the code for the first model. This Object will/can be used directly. The most important methods provided are train and predict.
The model is described in section 4.7 of the bachelor thesis.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import random
import pickle
import math
import gzip

from .pytorch_models import Model1
from .early_stopping import EarlyStopping
from .evaluation import Evaluation
from .data import Actor


def _print_loss_(in_=None, first=False, last=False):
    if in_ is not None:
        b, nb, l = in_
        print(" | {:>3d}/{:<3d} | {:>13.8f} |".format(b,nb,l))
    elif first is True:
        print(" > Train-Loop ...")
        print(" +---------+---------------+")
        print(" | {:^7s} | {:^13s} |".format("Batch","Batch-Loss"))
        print(" +---------+---------------+")
    elif last is True:
        print(" +---------+---------------+")
    return None


def _print_eval_(train=None, dev=None):
    print(" > Evaluation ...")
    print(" +-------+--------+--------+--------+--------+--------+--------+")
    print(" | {:5s} | {:6s} | {:6s} | {:6s} | {:6s} | {:6s} | {:6s} |".format("Set", "P", "R", "F1", "TPR", "TNR", "BA"))
    print(" +-------+--------+--------+--------+--------+--------+--------+")
    if train is not None:
        print(" | {:5s} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} |".format("train", train["p"], train["r"], train["f1"], train["tpr"], train["tnr"], train["ba"]))
    if dev is not None:
        print(" | {:5s} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f} |".format("dev", dev["p"], dev["r"], dev["f1"], dev["tpr"], dev["tnr"], dev["ba"]))
    print(" +-------+--------+--------+--------+--------+--------+--------+")


class Sampel(object):
    """
    Helper class.
    """
    def __init__(self, document, word_ids, char_ids, entity_spans, gold_actor_ids, gold_actor_spans):
        self.doc = document
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.entity_spans = entity_spans
        self.gold_actor_ids = gold_actor_ids
        self.gold_actor_spans = gold_actor_spans


class Step1(object):
    """
    This class implements the first model from the bachelor thesis (section 4.7).
    The public methods to use are:
        train
        predict
        estimate_delta_range
        evaluate_delta
    """
    def __init__(self, fasttext_embeddings, actor_embeddings,
                 word_vocab_size, char_vocab_size, actor_vocab_size, feature_vocab_size, char_pad_index, model_params,
                 update_weights=None, delta=None, model=None):
        # External data (input)
        self.fasttext_embeddings = fasttext_embeddings
        self.actor_embeddings = actor_embeddings
        # Fixed variabels determined by the given data
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.actor_vocab_size = actor_vocab_size
        self.feature_vocab_size = feature_vocab_size
        self.char_pad_index = char_pad_index
        # Model Hyper-Parameters (input)
        self.model_param = model_params
        self.lambda_ = model_params["lambda"]
        # Prepare vars for update weights
        self.update_weights = update_weights
        # Mode-Parameter (estimated during training)
        self.delta = delta
        # Pytorch model
        self.model = model
    
    def _pad_list_(self, list_):
        length = self.model_param["max_word_len"]
        if len(list_) > length:
            list_ = list_[:length]
        elif len(list_) < length:
            pad = length - len(list_)
            list_ = list_+(self.char_pad_index,)*pad
        return list_
    
    def _zero_(self, cuda=False):
        """
        Returns a zero tensor of shape (1,).
        If cuda is True, the tensor is on cuda.
        """
        z = torch.zeros(1)
        if cuda:
            z = z.cuda()
        return z
    
    def _generate_batches_(self, sampels, batch_size):
        """
        Creates a generator which returns batches sampeled from sampels with size of batch_size.
        """
        # calculate num batches
        n = len(sampels)
        num_batches = int(n/batch_size)
        # calculate num sampels to add to last batch to complete it
        r = n%batch_size
        r = batch_size-r if r > 0 else 0
        # randomize and create batches
        random.shuffle(sampels)
        all_batches = []
        for i in range(num_batches):
            start = int(batch_size*i)
            end = int(start+batch_size)
            all_batches.append(sampels[start:end])
        # if needed add sampels to last batch
        if r > 0:
            start = int(num_batches*batch_size)
            all_batches.append(sampels[start:]+sampels[:r])
        return all_batches
    
    def _sum_reduce_(self, bloss):
        """
        Reduces the list bloss by summing all contained tensors.
        """
        return torch.sum(torch.cat(bloss, dim=0),dim=0)
    
    def _create_sampels_(self, documents, device):
        """
        creates sample objects. converts data into tensors.
        """
        sampels = []
        with torch.no_grad():
            for doc in documents:
                # Create word- and char-id lists
                word_ids = list([t.i_fasttext for t in doc.tokens])
                char_ids = list([self._pad_list_(t.i_chars) for t in doc.tokens])
                # Create entity-span infos
                espans = []
                for entity in doc.entities:
                    begin,end = entity.span
                    # Create feature tensors
                    features = [entity.i_entity_class, entity.i_distance_feature]
                    features = torch.LongTensor(features).to(device=device)
                    espans.append( (begin, end, features) )
                espans = tuple(espans)
                # Create word- and char-id tensors
                word_ids = torch.LongTensor(word_ids).to(device=device)
                char_ids = torch.LongTensor(char_ids).to(device=device)
                # Create gold data
                gold_actors = {}
                for actor in doc.actors:
                    if actor.is_nil is False:
                        actor_idx = actor.i_wikidata_id
                        if actor_idx not in gold_actors:
                            gold_actors[actor_idx] = []
                        gold_actors[actor_idx].extend(actor.spans)
                if len(gold_actors) > 0:
                    gold_actor_idxs,gold_actor_spans = zip(*gold_actors.items())
                else:
                    gold_actor_idxs,gold_actor_spans = tuple(),tuple()
                #
                sampels.append(Sampel(doc, word_ids, char_ids, espans, gold_actor_idxs, gold_actor_spans))
        return sampels
    
    def _loss_(self, actor_scores, hidden, gold_actor_ids, gold_actor_spans):
        """
        Calculates the loss function for predicted actor_scores.
        The link_scores (hidden) is used if actor-spans are available
        """
        A,E = hidden.shape
        # prepare loss variables
        pos_loss = self._zero_(cuda=actor_scores.is_cuda)
        neg_loss = self._zero_(cuda=actor_scores.is_cuda)
        for i in range(A):
            if i in gold_actor_ids:
                # If the actor is gold ...
                aidx = gold_actor_ids.index(i)
                if len(gold_actor_spans[aidx]) > 0:
                    # and if updating the link score is possible
                    for eidx in gold_actor_spans[aidx]:
                        # Make positive update for the link score directly
                        p_score = hidden[i, eidx]
                        auxloss = torch.relu(self.lambda_-p_score).view(1)
                        pos_loss = pos_loss + auxloss
                else:
                    # otherwise make positive update for the link score via the score_1
                    p_score = actor_scores[i]
                    hloss = torch.relu(self.lambda_-p_score).view(1)
                    pos_loss = pos_loss + hloss
            else:
                # otherwise make negative update via score_1
                n_score = actor_scores[i]
                hloss = torch.relu(self.lambda_+n_score).view(1)
                neg_loss = neg_loss + hloss
        #
        loss = (pos_loss*self.update_weights[0] + neg_loss*self.update_weights[1]) / A
        return loss
    
    def _evaluate_(self, test_sampels, estimate_delta=False):
        """
        Calculates precision, recall and f1 for predicting the given sampels.
        If estimated_delta is True, the delta-parameter is estimated on the sampels, which is needed for prediction.
        """
        # Predict the scores for all test documents
        data = []
        with torch.no_grad():
            self.model.eval()
            for sampel in test_sampels:
                pred_scores,_ = self.model(sampel.word_ids,
                                           sampel.char_ids,
                                           sampel.entity_spans)
                data.append((pred_scores, sampel.gold_actor_ids))
            self.model.train()
        # Estimate delta (if specified)
        if estimate_delta is True:
            sorted_scores = {"pos":[], "neg":[]}
            for score1, g_actor_ids in data:
                for i in range(self.actor_vocab_size):
                    if i in g_actor_ids:
                        sorted_scores["pos"].append(score1[i].item())
                    else:
                        sorted_scores["neg"].append(score1[i].item())
            upper = np.quantile(sorted_scores["pos"], 0.5)
            lower = np.quantile(sorted_scores["neg"], 0.5)
            self.delta=(upper+lower)/2
            #self.delta=np.quantile(sorted_scores["pos"], 0.05)
        # Evaluate predicted actor-sets
        data2eval = []
        for score1, g_actor_ids in data:
            num_actors = score1.shape[0]
            y_hat = set(torch.arange(num_actors)[score1>self.delta].tolist())
            y = set(g_actor_ids)
            data2eval.append((y, y_hat))
        confusions,_ = Evaluation.confusion_matrix_by_label(data2eval, all_labels=set(range(self.actor_vocab_size)))
        confusions = Evaluation.micro_average_confusions(confusions)
        print(confusions)
        metrics = Evaluation.confusions_to_metrics(confusions)
        return metrics
    
    def estimate_delta_range(self, dev_docs, recall_ratio=1.0, precision_ratio=1.0, cuda=False):
        """
        This methods estimates the prediction-threshold delta_1 on the given development documents. 
        By default delta is estimated for max_precision and max_recall.
        Returns a tuple (max_recall_delta, max_precision_delta)
        """
        # Prepare pytorch cuda
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        # Init data
        test_sampels = self._create_sampels_(dev_docs, device=device)
        # Predict data and sort scores by pos-/neg-class
        sorted_scores = {"pos":[], "neg":[]}
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            for sampel in test_sampels:
                # Do a forward call
                actor_scores, _ = self.model(sampel.word_ids,
                                             sampel.char_ids,
                                             sampel.entity_spans)
                #
                for i in range(self.actor_vocab_size):
                    if i in sampel.gold_actor_ids:
                        sorted_scores["pos"].append(actor_scores[i].item())
                    else:
                        sorted_scores["neg"].append(actor_scores[i].item())
        #
        min_pos = np.quantile(sorted_scores["pos"], 1-recall_ratio)
        max_neg = np.quantile(sorted_scores["neg"], precision_ratio)
        delta_range = (min_pos, max_neg)
        return delta_range
    
    def evaluate_delta_range(self, delta_range, test_docs, num_steps=50, cuda=True):
        """
        Evaluates the model on differet values of the prediction-threshold. 
        num_steps many deltas ranging from delta_range[0], to delta_range[1] will be tested.
        The output is a list of dictionarys holding the a value of delta and some evaluation metrics.
        """
        # Prepare pytorch cuda
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        pred_data = list()
        with torch.no_grad():
            # Init data
            test_sampels = self._create_sampels_(test_docs, device=device)
            # predict scores 
            self.model.eval()
            for sampel in test_sampels:
                pred_scores,_ = self.model(sampel.word_ids,
                                           sampel.char_ids,
                                           sampel.entity_spans)
                pred_data.append((pred_scores, sampel.gold_actor_ids))
            # For each delta from the specified range....
            old_delta = self.delta  # store original delta value
            all_metrics = []
            for delta in np.linspace(delta_range[0], delta_range[1], num_steps):
                self.delta = delta
                # extract predictions using the current delta
                data2eval = []
                for score1, g_actor_ids in pred_data:
                    num_actors = score1.shape[0]
                    y_hat = set(torch.arange(num_actors)[score1>self.delta].tolist())
                    y = set(g_actor_ids)
                    data2eval.append((y, y_hat))
                # evaluate the predictions
                confusions,_ = Evaluation.confusion_matrix_by_label(data2eval)
                confusions = Evaluation.micro_average_confusions(confusions)
                metrics = Evaluation.confusions_to_metrics(confusions)
                metrics["delta"] = delta
                all_metrics.append(metrics)
            self.delta = old_delta   # restore original delta value
        return all_metrics
    
    def train(self, train_docs, dev_docs=None, num_epochs=100, batch_size=1, learning_rate=1.0, lr_reduction=0.5, patience=10, threshold=0.1, fine_tune=False, cuda=False):
        """
        Train this model on the train_docs and evaluate in each iteration on dev_docs.
        lr_reduction: number to be multiplied to the learning rate if no change was seen for some epochs
        patience: numer of epochs without change befor lr-reduction is applied
        threshold: threshold that must be overcome
        """
        # Prepare pytorch cuda
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        # Init (new) model
        if fine_tune is False:
            self.model = Model1(self.fasttext_embeddings, self.actor_embeddings, self.char_vocab_size, self.feature_vocab_size, self.char_pad_index,
                                efeature_embedding_dim=self.model_param["efeature_embedding_dim"],
                                char_embedding_dim=self.model_param["char_embedding_dim"],
                                filters_per_channel=self.model_param["filters_per_channel"],
                                kernels=self.model_param["kernels"],
                                num_lstm_layers=self.model_param["num_lstm_layers"],
                                ffnn_hidden_dim=self.model_param["ffnn_hidden_dim"],
                                num_ffnn_layers=self.model_param["num_ffnn_layers"],
                                embedding_dropout_p=self.model_param["embedding_dropout_p"],
                                hidden_dropout_p=self.model_param["hidden_dropout_p"],
                                use_context=self.model_param["use_context"],
                                context_size=self.model_param["context_size"],
                                K=self.model_param["K"])
        self.model.to(device)
        # Init optimizer and lr-scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", threshold_mode="rel", threshold=threshold, factor=lr_reduction, patience=patience, verbose=True)
        early_stop = EarlyStopping(patience=2*patience, delta=threshold, mode="rel")
        # Init data
        train_sampels = self._create_sampels_([d for d in train_docs], device=device)
        dev_sampels = self._create_sampels_([d for d in dev_docs], device=device)
        # Count Actor Doc-Frequencys
        actor_counts = [0,]*self.actor_vocab_size
        for s in train_sampels+dev_sampels:
            for aidx in s.gold_actor_ids:
                actor_counts[aidx] += 1
        # Set update-weights for positive and negative updates
        N = len(train_sampels+dev_sampels)
        if self.model_param["update_weights"] is True:
            #
            mean_doc_count = sum(actor_counts)/self.actor_vocab_size    ## len(
            inv_doc_count = N - mean_doc_count
            self.update_weights = [1/mean_doc_count, 1/inv_doc_count,]
            
        else:
            self.update_weights = [1, 1]
        # Init book-keeper
        bookkeeper = {"train":[],
                      "dev":[]}
        # Evaluate on data and estimated delta
        d_stats = self._evaluate_(dev_sampels, estimate_delta=True)
        t_stats = self._evaluate_(train_sampels)
        _print_eval_(train=t_stats, dev=d_stats)
        bookkeeper["train"].append(t_stats)
        bookkeeper["dev"].append(d_stats)
        # Train-Loop
        self.model.train()
        try:
            # In each epoch ...
            for e in range(num_epochs):
                print("\n\n > New epoch ({:>4d}/{:>4d})".format(e+1, num_epochs))
                # Randomly benerate batches
                batches = self._generate_batches_(train_sampels, batch_size)
                num_batches = len(batches)
                # For each batch ...
                _print_loss_(first=True)
                for b in range(num_batches):
                    # calculate loss for each batch-element
                    bloss = []
                    for s in range(batch_size):
                        sampel = batches[b][s]
                        # do a forward call
                        actor_scores, hidden = self.model(sampel.word_ids,
                                                          sampel.char_ids,
                                                          sampel.entity_spans)
                        # calculate loss
                        loss = self._loss_(actor_scores,
                                           hidden[0],        # is the Score_link
                                           sampel.gold_actor_ids,
                                           sampel.gold_actor_spans)
                        bloss.append(loss)
                    # reduce loss-values to batch-loss
                    bloss = self._sum_reduce_(bloss)
                    if bloss.requires_grad is False:
                        continue
                    # calculate gradients
                    optimizer.zero_grad()
                    bloss.backward()
                    # update parameteres
                    optimizer.step()
                    # print
                    _print_loss_(in_=(b+1, num_batches, bloss.item()))
                _print_loss_(last=True)
                # Estimate the hyper-parameter delta and
                # evaluate on data with estimated delta
                d_stats = self._evaluate_(dev_sampels, estimate_delta=True)
                t_stats = self._evaluate_(train_sampels)
                _print_eval_(train=t_stats, dev=d_stats)
                bookkeeper["train"].append(t_stats)
                bookkeeper["dev"].append(d_stats)
                print(" > estimated delta = {}".format(self.delta))
                # Schedule learning rate
                scheduler.step(d_stats["f1"])
                # Test for early stopping
                if early_stop.step(d_stats["f1"]) is True:
                    print(" > early stopped with F1 @ {:.2f} !!!".format(d_stats["f1"]))
                    print()
                    break
        except KeyboardInterrupt:
            print(" > user interrupt")
            # Make sure bookkeeper is consistent
            # (in case intterupt occures inbetween the evaluation)
            while len(bookkeeper["dev"]) > len(bookkeeper["train"]):
                del bookkeeper["dev"][-1]
            while len(bookkeeper["dev"]) < len(bookkeeper["train"]):
                del bookkeeper["train"][-1]
            # Make sure delta is a value
            if self.delta is None:
                print(" > WARNING: delta is None. Model is not ready to make predictions. The model must be evaulated at least once during training (run at least one epoch) to estimate this parameter .")
        return bookkeeper
    
    def predict(self, test_docs, cuda=False):
        """
        Predicts the given data. Stores the predicted actors into the specified document objects.
        """
        # Prepare pytorch cuda
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        # Init data
        test_sampels = self._create_sampels_(test_docs, device=device)
        #
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            for sampel in test_sampels:
                # Do a forward call
                actor_scores, hidden = self.model(sampel.word_ids,
                                                  sampel.char_ids,
                                                  sampel.entity_spans)
                # Select predicted actors
                pred_actors = torch.arange(actor_scores.shape[0])[actor_scores>self.delta].tolist()
                # Select predicted entity_spans
                pred_spans = []
                for aidx in pred_actors:
                    pidx = torch.argmax(hidden[0][aidx,:]).item()
                    pred_spans.append((sampel.entity_spans[pidx][0],sampel.entity_spans[pidx][1]))
                # Store predicted actors in document
                p_actors = [Actor(pi_wikidata_id=pred_actors[i], is_nil=False, p_spans=tuple([pred_spans[i],])) for i in range(len(pred_actors))]
                p_actors = tuple(sorted(p_actors, key=lambda x:x.pi_wikidata_id))
                sampel.doc.p_actors = p_actors
        return test_docs
    
    def store(self, path):
        """
        Stores this model to file.
        """
        model = (self.word_vocab_size,
                 self.char_vocab_size,
                 self.actor_vocab_size,
                 self.feature_vocab_size,
                 self.char_pad_index,
                 self.model_param,
                 self.update_weights,
                 self.delta,
                 self.model.state_dict(),
                 self.fasttext_embeddings.shape[1])
        with gzip.open(path, "w") as ofile:
            pickle.dump(model, ofile)
        return None
    
    @staticmethod
    def load(path):
        """
        Loads a model from file.
        """
        with gzip.open(path, "r") as ifile:
            model = pickle.load(ifile)
        #
        new_model = Model1(torch.empty(model[0], model[9]), torch.empty(model[2], model[9]), 
                           model[1], model[3], model[4],
                           efeature_embedding_dim=model[5]["efeature_embedding_dim"],
                           char_embedding_dim=model[5]["char_embedding_dim"],
                           filters_per_channel=model[5]["filters_per_channel"],
                           kernels=model[5]["kernels"],
                           num_lstm_layers=model[5]["num_lstm_layers"],
                           ffnn_hidden_dim=model[5]["ffnn_hidden_dim"],
                           num_ffnn_layers=model[5]["num_ffnn_layers"],
                           embedding_dropout_p=model[5]["embedding_dropout_p"],
                           hidden_dropout_p=model[5]["hidden_dropout_p"],
                           use_context=model[5]["use_context"],
                           context_size=model[5]["context_size"],
                           K=model[5]["K"])
        new_model.load_state_dict(model[8])
        #
        new_step = Step1(new_model.word_embedding.fast_text_embedding.weight.detach(), 
                         new_model.actor_embedding.weight.detach(), 
                         model[0], 
                         model[1], 
                         model[2], 
                         model[3], 
                         model[4],
                         model_params=model[5], 
                         update_weights=model[6], 
                         delta=model[7], 
                         model=new_model)
        #
        return new_step