#!/usr/bin/python3
"""
This file holds the code for the third model. This Object will/can be used directly. 
The most important methods provided are train and predict.
The model is described in section 4.9 of the bachelor thesis.

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

import matplotlib.pyplot as plt

from .pytorch_models import Model3
from .early_stopping import EarlyStopping
from .evaluation import Evaluation
from .data import Actor, Claim, StackedClaim


def _print_loss_(in_=None, first=False, last=False):
    if in_ is not None:
        b, nb, l = in_
        print(" | {:>3d}/{:<3d} | {:>11.6f} |".format(b,nb,l))
    elif first is True:
        print(" > Train-Loop ...")
        print(" +---------+-------------+")
        print(" | {:^7s} | {:^11s} |".format("Batch","Batch-Loss"))
        print(" +---------+-------------+")
    elif last is True:
        print(" +---------+-------------+")
    return None


def _print_eval_(dev=None, train=None, first=False, last=False):
        print(" > Evaluation ...")
        print(" +-------+--------+--------+--------+")
        print(" | {:5s} | {:6s} | {:6s} | {:6s} |".format("Set", "P", "R", "F1"))
        print(" +-------+--------+--------+--------+")
        print(" | {:5s} | {:6.2f} | {:6.2f} | {:6.2f} |".format("train", train["p"], train["r"], train["f1"]))
        print(" | {:5s} | {:6.2f} | {:6.2f} | {:6.2f} |".format("dev", dev["p"], dev["r"], dev["f1"]))
        print(" +-------+--------+--------+--------+")


class Sampel(object):
    """
    Helper class
    """
    def __init__(self, doc, word_ids, char_ids, claim_categories, attributed_actors, espans, cspans, dist_feats, gold, claim2update):
        self.doc = doc
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.claim_categories = claim_categories
        self.attributed_actors = attributed_actors
        self.espans = espans
        self.cspans = cspans
        self.distance_features = dist_feats
        self.gold = gold
        self.claim2update = claim2update


class Step3(object):
    """
    This class implements the third model from the bachelor thesis (section 4.9).
    The public methods to use are:
        train
        predict
        estimate_delta_ranges
        evaluate_delta_range
        evaluate_delta_nil_range
        store
        load
    """
    def __init__(self, fasttext_embeddings, actor_embeddings,
                 word_vocab_size, char_vocab_size, actor_vocab_size, entity_feature_vocab_size, claim_category_vocab_size, distance_feature_vocab_size, 
                 char_pad_idx, model_params, polarity_weights=None, delta=None, model=None):
        # External data (input)
        self.fasttext_embeddings = fasttext_embeddings
        self.actor_embeddings = actor_embeddings
        # Fixed variabels determined by the given data
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.actor_vocab_size = actor_vocab_size
        self.entity_feature_vocab_size = entity_feature_vocab_size
        self.claim_category_vocab_size = claim_category_vocab_size
        self.distance_feature_vocab_size = distance_feature_vocab_size
        self.char_pad_idx = char_pad_idx
        #
        self.model_param = model_params
        self.lambda_ = model_params["lambda"]
        self.polarity_weights = polarity_weights
        # Pytorch model
        self.delta = delta
        self.model = model
    
    def _pad_list_(self, list_):
        length = self.model_param["max_word_len"]
        if len(list_) > length:
            list_ = list_[:length]
        elif len(list_) < length:
            pad = length - len(list_)
            list_ = list_+(self.char_pad_idx,)*pad
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
    
    def _ones_(self, cuda=False):
        """
        Returns a one tensor of shape (2,).
        If cuda is True, the tensor is on cuda.
        """
        z = torch.zeros(2)
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
        # randomize and yield each batch
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
    
    def _create_sampels_(self, docs, expand=False, prediction=False, input_predictions=False, device="cpu"):
        """
        creates sample objects. converts data into tensors.
        For using the predicted input of model1 use the input_predicitions flag.
        """
        sampels = []
        with torch.no_grad():
            for doc in docs:
                 # Create word- and char-id tensor
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
                # Create tensors for word- and char-indices
                word_ids = torch.LongTensor(word_ids).to(device=device)
                char_ids = torch.LongTensor(char_ids).to(device=device)
                #
                dist_feats = torch.LongTensor(doc.i_distances).to(device)
                # Create data
                all_categories = []
                cspans = []
                gold_data = []
                attributed_actors = []
                # For each claim
                for i in range(len(doc.stacked_claims)):
                    ## Prepare Input-Data for this claim
                    # Collect span, category and attributed actors
                    begin, end = doc.stacked_claims[i].span
                    category_ids = doc.stacked_claims[i].i_categories
                    gold_actor_ids = [a.i_wikidata_id for a in doc.stacked_claims[i].actors if a.is_nil is False]
                    if input_predictions is True:
                        actor_ids = [a.pi_wikidata_id for a in doc.p_stacked_claims[i].p_actors if a.is_nil is False]
                    else:
                        actor_ids = gold_actor_ids
                    actor_ids = sorted(actor_ids)
                    ## Prepare gold data for the claim
                    if prediction is False:
                        # Create a tabel (actors x claim-cats) with each cell containing
                        # the id of the polarity connecting actor and category
                        gold_rels = [[None,]*len(category_ids),]*len(gold_actor_ids)
                        actor_spans = [set(),]*len(gold_actor_ids)
                        for c in doc.stacked_claims[i].unstacked_claims:
                            if c.actor.is_nil is False:
                                actor_idx = gold_actor_ids.index(c.actor.i_wikidata_id)
                                for category in c.i_categories:
                                    category_idx = category_ids.index(category)
                                    gold_rels[actor_idx][category_idx] = int(c.polarity)
                                actor_spans[actor_idx].update(c.actor.spans)
                        actor_spans = tuple([tuple(a) for a in actor_spans])
                        gold = (gold_actor_ids, gold_rels, actor_spans)
                    else:
                        gold = (None, None)
                    # Convert actor and category ids to tensor
                    category_ids = torch.LongTensor(category_ids).to(device)
                    actor_ids = torch.LongTensor(actor_ids).to(device)
                    #
                    gold_data.append(gold)
                    attributed_actors.append(actor_ids)
                    all_categories.append(category_ids)
                    cspans.append( (begin, end, category_ids) )
                # Create sampel for this document
                if expand is True:
                    for claim2update in range(len(doc.stacked_claims)):
                        sampels.append(Sampel(doc, word_ids, char_ids, all_categories, attributed_actors, espans, cspans, dist_feats, gold_data, claim2update))
                else:
                    sampels.append(Sampel(doc, word_ids, char_ids, all_categories, attributed_actors, espans, cspans, dist_feats, gold_data, None))
        print("Created {} sampels".format(len(sampels)))
        return sampels
    
    def _loss_(self, attributed_actors, claim_labels, p_scores, gold_data):
        """
        Calculates the loss function for predicted actor_attributions and nil_flags.
        """
        total_loss = self._zero_(attributed_actors[0].is_cuda)
        # For each Claim in Document
        num_claims = len(attributed_actors)
        for i in range(num_claims):
            loss = self._zero_(attributed_actors[0].is_cuda)
            # Gold-Data for that claim
            _, gold_relations, gold_actor_spans = gold_data[i]
            # Predicted scores for that claim
            score4, score_rel = p_scores[i]
            # If no actor from KB is attributed to the claim
            # skip this claim
            if score4.shape[0] == 0:
                continue
            # Input-Data for that claim
            actor_ids = attributed_actors[i]
            label_ids = claim_labels[i]
            # For each attributed actor
            num_actors = actor_ids.shape[0]
            num_labels = len(label_ids)
            for j in range(num_actors):
                g_actor_spans = gold_actor_spans[j]
                # If actor span is  available for this actor
                if len(g_actor_spans) > 0:
                    # For each attributed claim category
                    for k in range(num_labels):
                        gold_pol = gold_relations[j][k]
                        for eidx in range(score_rel.shape[1]):
                            violation = torch.relu(self.lambda_ - gold_pol*score_rel[j,eidx,k])
                            loss = loss + (violation*self.polarity_weights[gold_pol])
                else:
                    # For each attributed claim category
                    for k in range(num_labels):
                        gold_pol = gold_relations[j][k]
                        violation = torch.relu(self.lambda_ - gold_pol*score4[j,k])
                        loss = loss + (violation*self.polarity_weights[gold_pol])
            loss = loss / (num_actors*num_labels)
            total_loss = total_loss + loss
        return total_loss
    
    def _evaluate_(self, sampels, estimate_delta=False):
        """
        Calculates precision, recall and f1 for predicting the given sampels.
        If estimated_delta is True, the delta-parameter is estimated on the sampels, which is needed for prediction.
        """
        with torch.no_grad():
            self.model.eval()
            # Predict the test sampels
            pred_data = []
            for sampel in sampels:
                # Predict the scores for the sampels
                scores = self.model(sampel.word_ids,
                                    sampel.char_ids,
                                    sampel.attributed_actors,
                                    sampel.espans,
                                    sampel.cspans,
                                    sampel.distance_features)
                pred_data.append((sampel, scores))
            #
            pos_scores = []
            neg_scores = []
            if estimate_delta is True:
                for sampel, scores in pred_data:
                    for i in range(len(scores)):
                        score4, _ = scores[i]
                        num_actors = sampel.attributed_actors[i].shape[0]
                        num_cats =   sampel.claim_categories[i].shape[0]
                        _, gold_flag,_ = sampel.gold[i]
                        if num_actors==0:
                            continue
                        for j in range(num_actors):
                            for k in range(num_cats):
                                gold_rels = gold_flag[j][k]
                                pscore = score4[j,k].item()
                                if np.isnan(pscore):        ##debug remove
                                    print(pscore)
                                    input()
                                if gold_rels == 1:
                                    pos_scores.append(pscore)
                                elif gold_rels == -1:
                                    neg_scores.append(pscore)
                self.delta = (np.quantile(pos_scores, 0.5)+np.quantile(neg_scores, 0.5))/2
            #
            data2eval = []
            for sampel, scores in pred_data:
                # For each claim from the sampel
                num_claims = len(scores)
                for i in range(num_claims):
                    score4, _ = scores[i]
                    # If not actor from KB is attributed to the claim
                    # skip this claim
                    if score4.shape[0] == 0:
                        continue
                    gold_actor_ids, gold_rels, _ = sampel.gold[i]
                    pred_actor_ids = sampel.attributed_actors[i]
                    claim_cats = sampel.claim_categories[i]
                    num_cats = len(claim_cats)
                    # Extract predicted relations from predicted scores
                    y_hat = set()
                    for j in range(pred_actor_ids.shape[0]):
                        for k in range(num_cats):
                            polarity_scores = score4[j,k].item()
                            y_hat_p = 1 if polarity_scores > self.delta else -1
                            if y_hat_p == 1:  # TODO
                                y_hat_r = ("A{}".format(pred_actor_ids[j].item()), y_hat_p, "C{}".format(claim_cats[k].item()))
                                y_hat.add(y_hat_r)
                    # Create gold relations
                    y = set()
                    for j in range(len(gold_actor_ids)):
                        for k in range(num_cats):
                            y_p = gold_rels[j][k]
                            if y_p == 1:  # TODO
                                y_r = ("A{}".format(gold_actor_ids[j]), y_p, "C{}".format(claim_cats[k].item()))
                                y.add(y_r)
                    data2eval.append((y, y_hat))
            self.model.train()
        # Evaluate predictions
        res,_ = Evaluation.micro_multi_label_evaluation(data2eval) 
        print("Delta:", self.delta, "\n")
        return res
    
    def estimate_delta_range(self, dev_docs, recall_ratio=1.0, precision_ratio=1.0, cuda=False):
        """
        This methods estimates the prediction-threshold delta_4 on the given development documents. 
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
        pos_scores = []
        neg_scores = []
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            for sampel in test_sampels:
                # Predict the scores for the sampels
                scores = self.model(sampel.word_ids,
                                    sampel.char_ids,
                                    sampel.attributed_actors,
                                    sampel.espans,
                                    sampel.cspans,
                                    sampel.distance_features)
                #
                for i in range(len(scores)):
                    score4, _ = scores[i]
                    num_actors = sampel.attributed_actors[i].shape[0]
                    num_cats =   sampel.claim_categories[i].shape[0]
                    _,gold_flag,_ = sampel.gold[i]
                    if num_actors==0:
                        continue
                    for j in range(num_actors):
                        for k in range(num_cats):
                            gold_rels = gold_flag[j][k]
                            pscore = score4[j,k].item()
                            if np.isnan(pscore):        ##debug remove
                                print(pscore)
                                input()
                            if gold_rels == 1:
                                pos_scores.append(pscore)
                            elif gold_rels == -1:
                                neg_scores.append(pscore)
        #
        min_pos = np.quantile(pos_scores, 1-recall_ratio)
        max_neg = np.quantile(neg_scores, precision_ratio)
        delta_range = (min_pos, max_neg)
        return delta_range
    
    def evaluate_delta_range(self, delta_range, test_docs, input_predictions=False, num_steps=50, cuda=True):
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
        with torch.no_grad():
            # Init data
            test_sampels = self._create_sampels_(test_docs, input_predictions=input_predictions, device=device)
            # predict scores 
            self.model.eval()
            pred_data = []
            for sampel in test_sampels:
                # Predict the scores for the sampels
                scores = self.model(sampel.word_ids,
                                    sampel.char_ids,
                                    sampel.attributed_actors,
                                    sampel.espans,
                                    sampel.cspans,
                                    sampel.distance_features)
                pred_data.append((sampel, scores))
            # For each delta from the specified range....
            old_delta = self.delta  # store original delta value
            all_metrics = []
            for delta in np.linspace(delta_range[0], delta_range[1], num_steps):
                self.delta = delta
                data2eval = []
                for sampel, scores in pred_data:
                    # For each claim from the sampel
                    num_claims = len(scores)
                    for i in range(num_claims):
                        score4, score_rel = scores[i]
                        # If not actor from KB is attributed to the claim
                        # skip this claim
                        if score4.shape[0] == 0:
                            continue
                        gold_actor_ids, gold_rels, _ = sampel.gold[i]
                        pred_actor_ids = sampel.attributed_actors[i]
                        claim_cats = sampel.claim_categories[i]
                        num_cats = len(claim_cats)
                        # Extract predicted relations from predicted scores
                        y_hat = set()
                        for j in range(pred_actor_ids.shape[0]):
                            for k in range(num_cats):
                                polarity_scores = score4[j,k].item()
                                y_hat_p = 1 if polarity_scores > self.delta else -1
                                if y_hat_p == 1:  # TODO
                                    y_hat_r = ("A{}".format(pred_actor_ids[j].item()), y_hat_p, "C{}".format(claim_cats[k].item()))
                                    y_hat.add(y_hat_r)
                        # Create gold relations
                        y = set()
                        for j in range(len(gold_actor_ids)):
                            for k in range(num_cats):
                                y_p = gold_rels[j][k]
                                if y_p == 1:  # TODO
                                    y_r = ("A{}".format(gold_actor_ids[j]), y_p, "C{}".format(claim_cats[k].item()))
                                    y.add(y_r)
                        data2eval.append((y, y_hat))
                # evaluate the predictions
                metrics,_ = Evaluation.micro_multi_label_evaluation(data2eval)
                metrics["delta"] = delta
                all_metrics.append(metrics)
            self.delta = old_delta   # restore original delta value
        return all_metrics
    
    def train(self, train_docs, dev_docs, num_epochs=100, batch_size=1, learning_rate=1.0, lr_reduction=0.5, patience=10, threshold=0.1, fine_tune=False, cuda=False):
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
            self.model = Model3(self.fasttext_embeddings, self.actor_embeddings,
                                self.char_vocab_size,
                                self.entity_feature_vocab_size, self.claim_category_vocab_size, self.distance_feature_vocab_size, 
                                self.char_pad_idx,
                                efeature_embedding_dim=self.model_param["efeature_embedding_dim"],
                                cfeature_embedding_dim=self.model_param["cfeature_embedding_dim"],
                                dfeature_embedding_dim=self.model_param["dfeature_embedding_dim"],
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
                                K_context=self.model_param["K_context"],)
        self.model.to(device)
        self.model.train()
        # Init optimizers
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", threshold_mode="rel", threshold=threshold, factor=lr_reduction, patience=patience, verbose=True)
        early_stop = EarlyStopping(patience=2*patience, delta=threshold, mode="rel")
        # Init data
        train_sampels_unexpanded = self._create_sampels_([d for d in train_docs] , device=device)
        dev_sampels_unexpanded = self._create_sampels_([d for d in dev_docs], device=device)
        if self.model_param["update_claim_wise"] is True:
            train_sampels = self._create_sampels_([d for d in train_docs], expand=True, device=device)
        else:
            train_sampels = train_sampels_unexpanded
        # Calculate update weights for both polarities (inverse frequency)
        pol_counts = dict()
        for s in train_sampels_unexpanded+dev_sampels_unexpanded:
            for _, gold,_ in s.gold:
                for a in gold:
                    for cc in a:
                        p = cc
                        if p not in pol_counts:
                            pol_counts[p] = 0
                        pol_counts[p] += 1
        self.polarity_weights = dict([(k,1/math.sqrt(v)) for k,v in pol_counts.items()])
        print("PWs:", self.polarity_weights)
        #
        bookkeeper = {"train":[],
                      "dev":[]}
        # Evaluate untrained model
        dev_results = self._evaluate_(dev_sampels_unexpanded, estimate_delta=True)
        train_results = self._evaluate_(train_sampels_unexpanded)
        _print_eval_(train=train_results, dev=dev_results)
        bookkeeper["dev"].append(dev_results)
        bookkeeper["train"].append(train_results)
        # Train-Loop (A keyboard interrupt leads to return, 
        # the interrupt is NOT passed to next layer)
        try:
            # In each epoch ...
            for e in range(num_epochs):
                print("\n\n > new epoch started ({:>4d}/{:>4d})".format(e+1, num_epochs))
                # Randomly benerate batches
                batches = self._generate_batches_(train_sampels, batch_size)
                num_batches = len(batches)
                # For each batch ...
                _print_loss_(first=True)
                for b in range(num_batches):
                    # Calculate loss for each batch-element
                    bloss = []
                    for s in range(batch_size):
                        sampel = batches[b][s]
                        if self.model_param["update_claim_wise"] is True:
                            s = sampel.claim2update
                            e = s+1
                        else:
                            s = 0
                            e = len(sampel.cspans)
                        # Do a forward call
                        score4 = self.model(sampel.word_ids,
                                            sampel.char_ids,
                                            sampel.attributed_actors[s:e],
                                            sampel.espans,
                                            sampel.cspans[s:e],
                                            sampel.distance_features[:,s:e])
                        # Calculate loss
                        loss = self._loss_(sampel.attributed_actors[s:e],
                                           sampel.claim_categories[s:e],
                                           score4,
                                           sampel.gold[s:e])
                        bloss.append(loss)
                        #
                    # Reduce loss-values to batch-loss
                    bloss = self._sum_reduce_(bloss)
                    bloss = bloss
                    if bloss.requires_grad is False:
                        continue
                    # Calculate gradients
                    optimizer.zero_grad()
                    bloss.backward()
                    # Update parameteres
                    optimizer.step()
                    # Print loss
                    _print_loss_(in_=(b+1, num_batches, bloss.item()))
                _print_loss_(last=True)
                # Evaluate model
                dev_results = self._evaluate_(dev_sampels_unexpanded, estimate_delta=True)
                train_results = self._evaluate_(train_sampels_unexpanded)
                _print_eval_(train=train_results, dev=dev_results)
                bookkeeper["dev"].append(dev_results)
                bookkeeper["train"].append(train_results)
                # Schedule learning rate
                scheduler.step(dev_results["f1"])
                # Test for early stopping
                if early_stop.step(dev_results["f1"]) is True:
                    print(" > early stopped with F1 @ {:.2f} !!!".format(dev_results["f1"]))
                    print()
                    break
        except KeyboardInterrupt:
            print(" > user interrupt")
        print()
        return bookkeeper
    
    def predict(self, test_docs, input_predictions=False, cuda=False):
        """
        Predicts the given data. 
        Creates claim objects for each stacked claim and stores them in the document object.
        """
        # Prepare pytorch cuda settings
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            # Create Sampel objects from given test documents
            test_sampels = self._create_sampels_(test_docs, prediction=True, input_predictions=input_predictions, device=device)
            # For each sampel
            for sampel in test_sampels:
                # Predict scores for document
                pscores = self.model(sampel.word_ids,
                                     sampel.char_ids,
                                     sampel.attributed_actors,
                                     sampel.espans,
                                     sampel.cspans,
                                     sampel.distance_features)
                # Create StackedClaim objects if needed
                if sampel.doc.p_stacked_claims is None:
                    sampel.doc.p_stacked_claims = []
                    for gclaim in sampel.doc.stacked_claims:
                        new_claim = StackedClaim(anno_id=gclaim.anno_id,
                                             text=gclaim.text,
                                             span=gclaim.span,
                                             categories=gclaim.categories)
                        sampel.doc.p_stacked_claims.append(new_claim)
                    sampel.doc.p_stacked_claims = tuple(sampel.doc.p_stacked_claims)
                #
                
                num_claims = len(pscores)
                num_entities = len(sampel.cspans)
                num_polarities = 2
                predicted_claims = []
                for i in range(num_claims):
                    score4, score_rel = pscores[i]
                    if score4.shape[0] == 0:
                        sampel.doc.p_stacked_claims[i].p_unstacked_claims = tuple()
                    else:
                        # if the claim has an attributed actor from KB
                        # then extract predictions
                        actor_ids = sampel.attributed_actors[i]
                        cat_ids   = sampel.claim_categories[i]
                        num_actors     = actor_ids.shape[0]
                        num_categories = cat_ids.shape[0]
                        # create a list of predicted relations
                        # one for each claim category
                        p_relations = []
                        for j in range(num_actors):
                            actor_index = actor_ids[j].item()
                            for k in range(num_categories):
                                category_index = cat_ids[k].item()
                                # extract polarity and entity index from score rel
                                p_polarity = 1 if score4[j,k] > self.delta else -1
                                pred_rel = (actor_index, p_polarity, category_index)
                                p_relations.append(pred_rel)
                        # merge relations with same category
                        preds_by_actor_pol = {}
                        for p_rel in p_relations:
                            if p_rel[0:2] not in preds_by_actor_pol:
                                preds_by_actor_pol[p_rel[0:2]] = []
                            preds_by_actor_pol[p_rel[0:2]].append(p_rel)
                        # create unstacked claim objects from mergred relations
                        stacked_claim = sampel.doc.stacked_claims[i]
                        p_claims = []
                        for pair, p_rels in preds_by_actor_pol.items():
                            actor_idx, pol_idx = pair
                            p_cats = [e[2] for e in p_rels]
                            p_actor = Actor(pi_wikidata_id=actor_idx,
                                            is_nil=False)
                            p_claim = Claim(anno_id=stacked_claim.anno_id,
                                            text=stacked_claim.text,
                                            span=stacked_claim.span,
                                            p_actor=p_actor,
                                            pi_categories=tuple(sorted(p_cats)),
                                            p_polarity=str(pol_idx))
                            p_claims.append(p_claim)
                        sampel.doc.p_stacked_claims[i].p_unstacked_claims = tuple(sorted(p_claims, key=lambda x:x.span))
                        predicted_claims.extend(p_claims)
                # Store predicted claims in document
                predicted_claims = tuple(sorted(predicted_claims, key=lambda x:x.span))
                sampel.doc.p_claims = predicted_claims
        return test_docs
    
    def store(self, path):
        """
        Stores a model to the specified path
        """
        model = (self.word_vocab_size,
                 self.char_vocab_size,
                 self.actor_vocab_size,
                 self.entity_feature_vocab_size,
                 self.claim_category_vocab_size,
                 self.distance_feature_vocab_size,
                 self.char_pad_idx,
                 self.model_param,
                 self.polarity_weights,
                 self.delta,
                 self.model.state_dict(),
                 self.fasttext_embeddings.shape[1])
        with gzip.open(path, "w") as ofile:
            pickle.dump(model, ofile)
        return None
    
    @staticmethod
    def load(path):
        """
        Loads a model from the specified path
        """
        with gzip.open(path, "r") as ifile:
            model = pickle.load(ifile)
        #
        new_model = Model3(torch.empty(model[0], model[11]), torch.empty(model[2], model[11]),
                           model[1],
                           model[3], model[4], model[5], 
                           model[6],
                           efeature_embedding_dim=model[7]["efeature_embedding_dim"],
                           cfeature_embedding_dim=model[7]["cfeature_embedding_dim"],
                           dfeature_embedding_dim=model[7]["dfeature_embedding_dim"],
                           char_embedding_dim=model[7]["char_embedding_dim"],
                           filters_per_channel=model[7]["filters_per_channel"],
                           kernels=model[7]["kernels"],
                           num_lstm_layers=model[7]["num_lstm_layers"],
                           ffnn_hidden_dim=model[7]["ffnn_hidden_dim"],
                           num_ffnn_layers=model[7]["num_ffnn_layers"],
                           embedding_dropout_p=model[7]["embedding_dropout_p"],
                           hidden_dropout_p=model[7]["hidden_dropout_p"],
                           use_context=model[7]["use_context"],
                           context_size=model[7]["context_size"],
                           K_context=model[7]["K_context"],)
        new_model.load_state_dict(model[10])
        #
        new_step = Step3(new_model.word_embedding.fast_text_embedding.weight.detach(), 
                         new_model.actor_embedding.weight.detach(), 
                         model[0],
                         model[1],
                         model[2], 
                         model[3], 
                         model[4], 
                         model[5], 
                         model[6], 
                         model[7],
                         model[8],
                         model[9],
                         new_model)
        #
        return new_step