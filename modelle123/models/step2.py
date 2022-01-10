#!/usr/bin/python3
"""
This file holds the code for the second model. This Object will/can be used directly. 
The most important methods provided are train and predict.
The model is described in section 4.8 of the bachelor thesis.

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

from .pytorch_models import Model2
from .early_stopping import EarlyStopping
from .evaluation import Evaluation
from .data import Actor, StackedClaim



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


def _print_eval_(dev_attr=None, dev_nil=None, dev_nil_attr=None, train_attr=None, train_nil=None, train_nil_attr=None):
    print(" > Evaluation ...")
    print("          +--------------------------++--------------------------+---------------------------+")
    print("          | {:24s} || {:24s} || {:24s} |".format("Attribution", "Nil-Flags", "Nil-Attributions"))
    print(" +-------++--------+--------+--------++--------+--------+--------++--------+--------+--------+")
    print(" | {:5s} || {:6s} | {:6s} | {:6s} || {:6s} | {:6s} | {:6s} || {:6s} | {:6s} | {:6s} |".format("Set", "P", "R", "F1", "P", "R", "F1", "P", "R", "F1"))
    print(" +-------++--------+--------+--------++--------+--------+--------++--------+--------+--------+")
    if train_attr is not None:
        print(" | {:5s} || {:6.2f} | {:6.2f} | {:6.2f} || {:6.2f} | {:6.2f} | {:6.2f} || {:6.2f} | {:6.2f} | {:6.2f} |".format("train", train_attr["p"], train_attr["r"], train_attr["f1"], train_nil["p"], train_nil["r"], train_nil["f1"], train_nil_attr["p"], train_nil_attr["r"], train_nil_attr["f1"]))
    if dev_attr is not None:
        print(" | {:5s} || {:6.2f} | {:6.2f} | {:6.2f} || {:6.2f} | {:6.2f} | {:6.2f} || {:6.2f} | {:6.2f} | {:6.2f} |".format("dev", dev_attr["p"], dev_attr["r"], dev_attr["f1"], dev_nil["p"], dev_nil["r"], dev_nil["f1"], dev_nil_attr["p"], dev_nil_attr["r"], dev_nil_attr["f1"]))
    print(" +-------++--------+--------+--------++--------+--------+--------++--------+--------+--------+")


class Sampel(object):
    """
    Helper class
    """
    def __init__(self, doc, actor_candidates, word_ids, char_ids, espans, cspans, dist_feats, claim_to_update, gold_attr, gold_spans, gold_nil_attr, gold_nil_spans, gold_nil_flags):
        self.doc = doc
        #
        self.claim_to_update = claim_to_update
        self.actor_candidates = actor_candidates
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.entity_spans = espans
        self.claim_spans = cspans
        self.distance_features = dist_feats
        #
        self.g_attributions = gold_attr
        self.g_actor_spans = gold_spans
        #
        self.g_nil_attributions = gold_nil_attr
        self.g_nil_spans = gold_nil_spans
        #
        self.g_nil_flags = gold_nil_flags


class Step2(object):
    """
    This class implements the second model from the bachelor thesis (section 4.8).
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
                 word_vocab_size, char_vocab_size, actor_vocab_size, 
                 entity_feature_vocab_size, claim_feature_vocab_size, dist_feature_vocab_size, 
                 char_pad_index,
                 model_params, delta=None, delta_nil=None, model=None):
        # External data (input)
        self.fasttext_embeddings = fasttext_embeddings
        self.actor_embeddings = actor_embeddings
        # Fixed variabels determined by the given data
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.actor_vocab_size = actor_vocab_size
        self.entity_feature_vocab_size = entity_feature_vocab_size
        self.claim_feature_vocab_size = claim_feature_vocab_size
        self.dist_feature_vocab_size = dist_feature_vocab_size
        self.char_pad_index=char_pad_index
        # Model Hyper-Parameters (input)
        self.model_param = model_params
        self.lambda_ = model_params["lambda"]
        self.lambda_nil = model_params["lambda_nil"]
        self.num_nils_to_take = 1
        self.loss_weight = model_params["loss_weights"]
        # Mode-Parameter (estimated during training)
        self.delta = delta
        self.delta_nil = delta_nil
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
    
    def _create_sampels_(self, docs, expand=False, input_predictions=False, device="cpu"):
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
                # Create gold information and claims-span infos
                cspans = []
                gold_attr = []
                gold_nil_attr = []
                gold_nil_flags = []
                gold_nil_spans = []
                gold_spans = []
                for c in doc.stacked_claims:
                    begin,end = c.span
                    feats = torch.LongTensor(c.i_categories).to(device)
                    cspans.append( (begin, end, feats) )
                    # Create actor-gold data
                    gold_actors = {}
                    for actor in c.actors:
                        if actor.is_nil is False:
                            actor_idx = actor.i_wikidata_id
                            if actor_idx not in gold_actors:
                                gold_actors[actor_idx] = []
                            gold_actors[actor_idx].extend(actor.spans)
                    if len(gold_actors) > 0:
                        actor_idxs,actor_spans = zip(*gold_actors.items())
                    else:
                        actor_idxs,actor_spans = tuple(),tuple()
                    actor_idxs = sorted(actor_idxs)
                    gold_attr.append(tuple(actor_idxs))
                    gold_spans.append(tuple(actor_spans))
                    # Create nil-actor-gold data
                    gold_actors = {}
                    for actor in c.actors:
                        if actor.is_nil is True:
                            actor_name = actor.normal_name
                            if actor_name not in gold_actors:
                                gold_actors[actor_name] = []
                            gold_actors[actor_name].extend(actor.spans)
                    if len(gold_actors) > 0:
                        actor_names,actor_spans = zip(*gold_actors.items())
                    else:
                        actor_names,actor_spans = tuple(),tuple()
                    actor_names = sorted(actor_names)
                    gold_nil_attr.append(tuple(actor_names))
                    gold_nil_spans.append(tuple(actor_spans))
                    # Create gold nil-flag data
                    flag = 1 if len(actor_names)>0 else -1
                    gold_nil_flags.append(flag)
                    
                #
                cspans = tuple(cspans)
                gold_attr = tuple(gold_attr)
                gold_spans = tuple(gold_spans)
                gold_nil_attr = tuple(gold_nil_attr)
                gold_nil_spans = tuple(gold_nil_spans)
                gold_nil_flags = tuple(gold_nil_flags)
                #
                dist_feats = torch.LongTensor(doc.i_distances).to(device)
                # Create actor doc set
                if input_predictions is True:
                    actor_candidates = set([a.pi_wikidata_id for a in doc.p_actors if a.is_nil is False])
                else:
                    actor_candidates = set([a.i_wikidata_id for a in doc.actors if a.is_nil is False])
                actor_candidates = sorted(actor_candidates)
                actor_candidates = torch.LongTensor(actor_candidates).to(device)
                #
                if expand is True:
                    for i in range(len(doc.stacked_claims)):
                        sampels.append(Sampel(doc, actor_candidates, word_ids, char_ids, espans, cspans, dist_feats, i,
                                              gold_attr, gold_spans, gold_nil_attr, gold_nil_spans, gold_nil_flags))
                else:
                    sampels.append(Sampel(doc, actor_candidates, word_ids, char_ids, espans, cspans, dist_feats, None,
                                          gold_attr, gold_spans, gold_nil_attr, gold_nil_spans, gold_nil_flags))
        return sampels
    
    def _loss_(self, actor_candidates, score2, score_nil, hidden, gold_attribution, gold_attr_spans, gold_nil_attr_spans, gold_nil_flags):
        """
        Calculates the loss function for predicted actor_attributions and nil_flags.
        """
        A,C = score2.shape
        #
        v1 = self._zero_(actor_candidates.is_cuda)
        v2 = self._zero_(actor_candidates.is_cuda)
        #
        score_linked_attr, score_nil_attr = hidden
        #
        #
        for i in range(C):
            gold_actors = gold_attribution[i]
            g_actor_spans = gold_attr_spans[i]
            # For each actor in the candidate set
            for j in range(A):
                aidx = actor_candidates[j].item()
                if aidx in gold_actors:
                    aidx_in_gold = gold_actors.index(aidx)
                    available_spans = g_actor_spans[aidx_in_gold]
                    if len(available_spans)>0:
                        # If the actor j should be attributed to the claim i via entity k
                        for k in available_spans:
                            pos_score = score_linked_attr[j,k,i]
                            v1 = v1 + torch.relu(self.lambda_ - pos_score)
                    else:
                        # If the actor j should be attributed to the claim i
                        pos_score = score2[j,i]
                        v1 = v1 + torch.relu(self.lambda_ - pos_score)
                else:
                    # If the actor j should not be attributed to claim i
                    neg_score = score2[j,i]
                    v1 = v1 + torch.relu(self.lambda_ + neg_score)
            # If a nil_actor is attributed
            if gold_nil_flags[i] == 1:
                # Update the (attribution score via the) nil_score
                v2 = v2 + torch.relu(self.lambda_nil - score_nil[i])
            else:
                # If no nil_actor is attributed then
                # update the (attribution score via the) nil_score
                v2 = v2 + torch.relu(self.lambda_nil + score_nil[i])
        if A > 0:
            v1 = v1*(1/A)
        v1 = v1 / C
        v2 = v2 / C
        loss_val = v1*self.loss_weight[0] + v2*self.loss_weight[1]
        return loss_val
    
    def _evaluate_(self, sampels, estimate_delta=False):
        """
        Calculates precision, recall and f1 for predicting the given sampels.
        If estimated_delta is True, the delta-parameter is estimated on the sampels, which is needed for prediction.
        """
        data = []
        with torch.no_grad():
            # Calculate scores for test data
            self.model.eval()
            for s in sampels:
                sampel = s
                score2, nil_scores, hidden = self.model(sampel.word_ids,
                                                        sampel.char_ids,
                                                        sampel.actor_candidates,
                                                        sampel.entity_spans,
                                                        sampel.claim_spans,
                                                        sampel.distance_features)
                data.append( (sampel, score2, nil_scores, hidden) )
            self.model.train()
            # Estimate the delta threshold needed for prediction
            if estimate_delta is True:
                # sort all predicted score into gold and not-gold
                sorted_score2 = {"pos":[], "neg":[]}
                sorted_score_nil = {"pos":[], "neg":[]}
                for sampel, score2, score_nil, _ in data:
                    num_claims = score2.shape[1]
                    num_actors = score2.shape[0]
                    for i in range(num_claims):
                        # sort score_nil
                        if sampel.g_nil_flags[i] == 1:
                            sorted_score_nil["pos"].append(score_nil[i].item())
                        else:
                            sorted_score_nil["neg"].append(score_nil[i].item())
                        # sort score2
                        for j in range(num_actors):
                            aj = sampel.actor_candidates[j].item()
                            if aj in sampel.g_attributions[i]:
                                sorted_score2["pos"].append(score2[j, i].item())
                            else:
                                sorted_score2["neg"].append(score2[j, i].item())
                # Estimate thresolds as the middle between medians of gold and not-gold scores
                upper = np.quantile(sorted_score2["pos"], 0.5)
                lower = np.quantile(sorted_score2["neg"], 0.5)
                self.delta=(upper+lower)/2
                #
                upper = np.quantile(sorted_score_nil["pos"], 0.5)
                lower = np.quantile(sorted_score_nil["neg"], 0.5)
                self.delta_nil=(upper+lower)/2
            # Predict actor-claim attributions
            data2eval = []
            for sampel,score2,_,_ in data:
                num_actors = score2.shape[0]
                num_claims = score2.shape[1]
                actor_ids = sampel.actor_candidates
                for i in range(num_claims):
                    pred_scores = score2[:,i]
                    pred_indices = torch.arange(num_actors)[pred_scores>self.delta]
                    y_hat = actor_ids[pred_indices].tolist()
                    y = sampel.g_attributions[i]
                    data2eval.append((y, y_hat))
            # Evaluate attributions
            confusions,_ = Evaluation.confusion_matrix_by_label(data2eval, set(range(self.actor_vocab_size)))
            confusions = Evaluation.micro_average_confusions(confusions)
            print(confusions)
            attribution_res = Evaluation.confusions_to_metrics(confusions)
            # Predict nil flags for each claim
            data2eval = []
            for sampel,_,score_nil,_ in data:
                num_claims = score_nil.shape[0]
                pred_flags = torch.where(score_nil>self.delta_nil, 1, -1).tolist()
                for i in range(num_claims):
                    y_hat = pred_flags[i]
                    y = sampel.g_nil_flags[i]
                    data2eval.append((y, y_hat))
            # Evaluate nil flags
            nil_flag_res = Evaluation.binary_evaluation(data2eval)
            # Predict names of nil_actors
            data2eval = []
            #print("NIL-Claim-Attribution")
            for sampel,_ ,score_nil, hidden in data:
                score_nil_attr = hidden[1]
                num_espans = score_nil_attr.shape[0]
                num_claims = score_nil_attr.shape[1]
                pred_flags = torch.where(score_nil>self.delta_nil, 1, -1).tolist()
                for i in range(num_claims):
                    if pred_flags[i] == 1:
                        # if existance of nil is predicted positive
                        # use the nil_attr_score to select an entity-span as the nil 
                        pred_scores = score_nil_attr[:, i]
                        pred_indices = torch.arange(num_espans)[torch.argsort(pred_scores)].tolist()
                        pred_indices = pred_indices[-self.num_nils_to_take:]
                        pred_spans = [sampel.entity_spans[idx] for idx in pred_indices]
                        # create the name of the selected nil-actor from tokens (without any normalization)
                        y_hat = [" ".join([t.text for t in sampel.doc.tokens[s[0]:s[1]]]) for s in pred_spans]
                    else:
                        y_hat = list()
                    y = sampel.g_nil_attributions[i]
                    data2eval.append((y, y_hat))
            # Evaluate names of nil actors
            confusions,_ = Evaluation.confusion_matrix_by_label(data2eval)
            confusions = Evaluation.micro_average_confusions(confusions)
            nil_attribution_res = Evaluation.confusions_to_metrics(confusions)
        # return
        return attribution_res, nil_flag_res, nil_attribution_res
    
    def estimate_delta_ranges(self, dev_docs, recall_ratio=1.0, precision_ratio=1.0, cuda=False):
        """
        This methods estimates the prediction-thresholds delta_2 and delta_3 on the given development documents. 
        By default delta is estimated for max_precision and max_recall.
        Returns a two tuples (max_recall_delta_2, max_precision_delta_2), (max_recall_delta_3, max_precision_delta_3)
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
        #
        sorted_score2 = {"pos":[], "neg":[]}
        sorted_score_nil = {"pos":[], "neg":[]}
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            # sort all predicted score into gold and not-gold
            for sampel in test_sampels:
                # Do a forward call
                score2, score_nil, _ = self.model(sampel.word_ids,
                                                   sampel.char_ids,
                                                   sampel.actor_candidates,
                                                   sampel.entity_spans,
                                                   sampel.claim_spans,
                                                   sampel.distance_features)
                #
                num_claims = score2.shape[1]
                num_actors = score2.shape[0]
                for i in range(num_claims):
                    # sort score_nil
                    if sampel.g_nil_flags[i] == 1:
                        sorted_score_nil["pos"].append(score_nil[i].item())
                    else:
                        sorted_score_nil["neg"].append(score_nil[i].item())
                    # sort score2
                    for j in range(num_actors):
                        aj = sampel.actor_candidates[j].item()
                        if aj in sampel.g_attributions[i]:
                            sorted_score2["pos"].append(score2[j, i].item())
                        else:
                            sorted_score2["neg"].append(score2[j, i].item())
        #
        max_r = np.quantile(sorted_score2["pos"], 1-recall_ratio)
        max_p = np.quantile(sorted_score2["neg"], precision_ratio)
        delta_range = (max_r, max_p)
        #
        max_r = np.quantile(sorted_score_nil["pos"], 1-recall_ratio)
        max_p = np.quantile(sorted_score_nil["neg"], precision_ratio)
        delta_nil_range = (max_r, max_p)
        #
        return delta_range, delta_nil_range
    
    def evaluate_delta_range(self, delta_range, test_docs, input_predictions=False, num_steps=50, cuda=True):
        """
        Evaluates the model on differet values of the prediction-threshold delta_2. 
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
        # Predict scores
        pred_data = []
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            test_sampels = self._create_sampels_(test_docs, input_predictions=input_predictions, device=device)
            for sampel in test_sampels:
                document = sampel.doc
                # predict the document
                score2, score_nil, hidden = self.model(sampel.word_ids,
                                                       sampel.char_ids,
                                                       sampel.actor_candidates,
                                                       sampel.entity_spans,
                                                       sampel.claim_spans,
                                                       sampel.distance_features)
                pred_data.append((sampel, score2))
        # Extract predictions from scores for diffrente values of delta
        old_delta = self.delta
        all_metrics = []
        for delta in np.linspace(delta_range[0], delta_range[1], num_steps):
            self.delta = delta
            # Extract predictions
            data2eval = []
            for sampel, score2 in pred_data:
                num_actors = score2.shape[0]
                num_claims = score2.shape[1]
                actor_ids = sampel.actor_candidates
                for i in range(num_claims):
                    pred_scores = score2[:,i]
                    pred_indices = torch.arange(num_actors)[pred_scores>self.delta]
                    y_hat = actor_ids[pred_indices].tolist()
                    y = sampel.g_attributions[i]
                    data2eval.append((y, y_hat))
            # Evaluate attributions
            confusions,_ = Evaluation.confusion_matrix_by_label(data2eval)
            confusions = Evaluation.micro_average_confusions(confusions)
            metrics = Evaluation.confusions_to_metrics(confusions)
            metrics["delta"] = delta
            all_metrics.append(metrics)
        #
        self.delta = old_delta
        return all_metrics
    
    def evaluate_delta_nil_range(self, delta_range, test_docs, input_predictions=False, num_steps=50, cuda=True):
        """
        Evaluates the model on differet values of the prediction-threshold delta_3. 
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
        # Predict scores
        pred_data = []
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            test_sampels = self._create_sampels_(test_docs, input_predictions=input_predictions, device=device)
            for sampel in test_sampels:
                document = sampel.doc
                # predict the document
                score2, score_nil, hidden = self.model(sampel.word_ids,
                                                       sampel.char_ids,
                                                       sampel.actor_candidates,
                                                       sampel.entity_spans,
                                                       sampel.claim_spans,
                                                       sampel.distance_features)
                pred_data.append((sampel, score_nil, hidden))
        # Extract predictions from scores for diffrente values of delta
        old_delta_nil = self.delta_nil
        all_attr_metrics = []
        all_flag_metrics = []
        for delta in np.linspace(delta_range[0], delta_range[1], num_steps):
            self.delta_nil = delta
            # Extract predicted nil flags for each claim
            data2eval = []
            for sampel,score_nil,_ in pred_data:
                num_claims = score_nil.shape[0]
                pred_flags = torch.where(score_nil>self.delta_nil, 1, -1).tolist()
                for i in range(num_claims):
                    y_hat = pred_flags[i]
                    y = sampel.g_nil_flags[i]
                    data2eval.append((y, y_hat))
            # Evaluate nil flags
            metrics = Evaluation.binary_evaluation(data2eval)
            metrics["delta"] = delta
            all_flag_metrics.append(metrics)
            # Extract predicted names of nil_actors
            data2eval = []
            for sampel,score_nil, hidden in pred_data:
                score_nil_attr = hidden[1]
                num_espans = score_nil_attr.shape[0]
                num_claims = score_nil_attr.shape[1]
                pred_flags = torch.where(score_nil>self.delta_nil, 1, -1).tolist()
                for i in range(num_claims):
                    if pred_flags[i] == 1:
                        # if existance of nil is predicted positive
                        # use the nil_attr_score to select an entity-span as the nil 
                        pred_scores = score_nil_attr[:, i]
                        pred_indices = torch.arange(num_espans)[torch.argsort(pred_scores)].tolist()
                        pred_indices = pred_indices[-self.num_nils_to_take:]
                        pred_spans = [sampel.entity_spans[idx] for idx in pred_indices]
                        # create the name of the selected nil-actor from tokens (without any normalization)
                        y_hat = [" ".join([t.text for t in sampel.doc.tokens[s[0]:s[1]]]) for s in pred_spans]
                    else:
                        y_hat = list()
                    y = sampel.g_nil_attributions[i]
                    data2eval.append((y, y_hat))
            # Evaluate names of nil actors
            confusions,_ = Evaluation.confusion_matrix_by_label(data2eval)
            confusions = Evaluation.micro_average_confusions(confusions)
            metrics = Evaluation.confusions_to_metrics(confusions)
            metrics["delta"] = delta
            all_attr_metrics.append(metrics)
        #
        self.delta_nil = old_delta_nil
        return all_flag_metrics, all_attr_metrics
    
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
            self.model = Model2(self.fasttext_embeddings, self.actor_embeddings,
                                self.char_vocab_size, self.char_pad_index,
                                self.entity_feature_vocab_size, self.claim_feature_vocab_size, self.dist_feature_vocab_size,
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
                                K_context=self.model_param["K_context"])
        self.model.to(device)
        self.model.train()
        # Init optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", threshold_mode="rel", threshold=threshold, factor=lr_reduction, patience=patience, verbose=True)
        early_stop = EarlyStopping(patience=2*patience, delta=threshold, mode="rel")
        # Init data
        train_sampels_unexpanded = self._create_sampels_([d for d in train_docs], device=device)
        dev_sampels_unexpanded = self._create_sampels_([d for d in dev_docs], device=device)
        if self.model_param["update_claim_wise"] is True:
            train_sampels = self._create_sampels_([d for d in train_docs], expand=True, device=device)
        else:
            train_sampels = train_sampels_unexpanded
        print("Created {} train instance".format(len(train_sampels)))
        # Init book-keeper
        bookkeeper = {"train":{"attr":[], "nil":[], "nil_heur":[]},
                      "dev":{"attr":[], "nil":[], "nil_heur":[]}}
        # Estimate the hyper-parameter delta and
        # Evaluate on train-data with estimated delta
        dev_attr, dev_nil, dev_nil_attr = self._evaluate_(dev_sampels_unexpanded, estimate_delta=True)
        train_attr, train_nil, train_nil_attr = self._evaluate_(train_sampels_unexpanded)
        _print_eval_(train_attr=train_attr, train_nil=train_nil, train_nil_attr=train_nil_attr,
                     dev_attr=dev_attr, dev_nil=dev_nil, dev_nil_attr=dev_nil_attr)
        bookkeeper["train"]["attr"].append(train_attr)
        bookkeeper["train"]["nil"].append(train_nil)
        bookkeeper["train"]["nil_heur"].append(train_nil_attr)
        bookkeeper["dev"]["attr"].append(dev_attr)
        bookkeeper["dev"]["nil"].append(dev_nil)
        bookkeeper["dev"]["nil_heur"].append(dev_nil_attr)
        print(" > estimated delta = {}".format(self.delta))
        print(" > estimated delta-nil = {}".format(self.delta_nil))
        # Train-Loop (A keyboard interrupt leads to return, 
        # the interrupt is NOT passed to next layer)
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
                        if self.model_param["update_claim_wise"] is True:
                            s = sampel.claim_to_update
                            e = s+1
                        else:
                            s = 0
                            e = len(sampel.claim_spans)
                        score_2, nil_scores, hidden = self.model(sampel.word_ids,
                                                                 sampel.char_ids,
                                                                 sampel.actor_candidates,
                                                                 sampel.entity_spans,
                                                                 sampel.claim_spans[s:e],
                                                                 sampel.distance_features[:, s:e])
                        # calculate loss
                        loss = self._loss_(sampel.actor_candidates,
                                           score_2,
                                           nil_scores,
                                           hidden,
                                           sampel.g_attributions[s:e],
                                           sampel.g_actor_spans[s:e],
                                           sampel.g_nil_spans[s:e],
                                           sampel.g_nil_flags[s:e])
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
                # Evaluate on train-data with estimated delta
                dev_attr, dev_nil, dev_nil_attr = self._evaluate_(dev_sampels_unexpanded, estimate_delta=True)
                train_attr, train_nil, train_nil_attr = self._evaluate_(train_sampels_unexpanded)
                _print_eval_(train_attr=train_attr, train_nil=train_nil, train_nil_attr=train_nil_attr,
                             dev_attr=dev_attr, dev_nil=dev_nil, dev_nil_attr=dev_nil_attr)
                bookkeeper["train"]["attr"].append(train_attr)
                bookkeeper["train"]["nil"].append(train_nil)
                bookkeeper["train"]["nil_heur"].append(train_nil_attr)
                bookkeeper["dev"]["attr"].append(dev_attr)
                bookkeeper["dev"]["nil"].append(dev_nil)
                bookkeeper["dev"]["nil_heur"].append(dev_nil_attr)
                print(" > estimated delta = {}".format(self.delta))
                print(" > estimated delta-nil = {}".format(self.delta_nil))
                # Schedule learning rate
                scheduler.step(dev_attr["f1"]+dev_nil["f1"])
                # Test for early stopping
                if early_stop.step(dev_attr["f1"]+dev_nil["f1"]) is True:
                    print(" > early stopped with attribution-F1 @ {:.2f} !!!".format(dev_attr["f1"]))
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
    
    def predict(self, test_docs, input_predictions=False, cuda=False):
        """
        Predicts the given data. 
        Creates stacked claim objects and stores them in the documents.
        """
        # Prepare pytorch cuda
        if cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device =  torch.device("cpu")
            if cuda:
                print(" > WARNING: CUDA is not available. CPU is used instead.")
        # Predict
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            test_sampels = self._create_sampels_(test_docs, input_predictions=input_predictions, device=device)
            for sampel in test_sampels:
                document = sampel.doc
                # predict the document
                score2, score_nil, hidden = self.model(sampel.word_ids,
                                                       sampel.char_ids,
                                                       sampel.actor_candidates,
                                                       sampel.entity_spans,
                                                       sampel.claim_spans,
                                                       sampel.distance_features)
                link_attr_scores, nil_attr_scores = hidden
                num_actors = score2.shape[0]
                num_claims = score2.shape[1]
                # if no predicted claims exists
                if document.p_stacked_claims is None:
                    document.p_stacked_claims = []
                    # create a claim object for each gold-claim
                    for i in range(num_claims):
                        claim = StackedClaim(anno_id=document.stacked_claims[i].anno_id,
                                             text=document.stacked_claims[i].text,
                                             span=document.stacked_claims[i].span,
                                             categories=document.stacked_claims[i].categories)
                        document.p_stacked_claims.append(claim)
                    document.p_stacked_claims = tuple(document.p_stacked_claims)
                # if there is any actor in the candidate set
                if sampel.actor_candidates.shape[0] == 0:
                    # predict nil actors for all claims
                    for i in range(num_claims):
                        # predict nil entities
                        nil_indices = torch.argsort(nil_attr_scores[:,i]).tolist()[-self.num_nils_to_take:]
                        nil_entities = [document.entities[nidx] for nidx in nil_indices]
                        nil_actors = [Actor(p_spans=tuple([e.span,]),
                                            p_normal_name=" ".join([t.text for t in document.tokens[e.span[0]:e.span[1]]]), 
                                            is_nil=True) for e in nil_entities]
                        nil_actors = tuple(sorted(nil_actors, key=lambda x:x.p_spans))
                        document.p_stacked_claims[i].p_actors = nil_actors
                else:
                # if at least one actor is in the candidate set
                    # extract predicted nil-flags
                    pred_nil_flags = [score_nil[i]>self.delta_nil for i in range(num_claims)]
                    for i in range(num_claims):
                        # extract predicted attributions
                        actor_indices = torch.arange(num_actors)[score2[:,i]>self.delta]
                        actor_entity_indices = torch.argmax(link_attr_scores[actor_indices, :, i], dim=1).tolist()
                        actor_indices = actor_indices.tolist()
                        pred_actors = [Actor(p_spans=tuple([document.entities[k].span, ]),
                                             pi_wikidata_id = sampel.actor_candidates[j].item(),
                                             is_nil=False) for j,k in zip(actor_indices,actor_entity_indices)]
                        # extract predicted nil-attributions
                        if pred_nil_flags[i] == 1:
                            nil_indices = torch.argsort(nil_attr_scores[:,i]).tolist()[-self.num_nils_to_take:]
                            nil_entities = [document.entities[nidx] for nidx in nil_indices]
                            nil_actors = [Actor(p_spans=tuple([e.span,]),
                                                p_normal_name=" ".join([t.text for t in document.tokens[e.span[0]:e.span[1]]]), 
                                                is_nil=True) for e in nil_entities]
                            pred_actors += nil_actors
                        pred_actors = tuple(sorted(pred_actors, key=lambda x:x.p_spans))
                        document.p_stacked_claims[i].p_actors = pred_actors
        return test_docs
    
    def store(self, path):
        """
        Store this model to the specifed location
        """
        model = (self.word_vocab_size,
                 self.char_vocab_size,
                 self.actor_vocab_size, 
                 self.entity_feature_vocab_size,
                 self.claim_feature_vocab_size,
                 self.dist_feature_vocab_size, 
                 self.char_pad_index,
                 self.model_param, 
                 self.delta,
                 self.delta_nil, 
                 self.model.state_dict(),
                 self.fasttext_embeddings.shape[1])
        with gzip.open(path, "w") as ofile:
            pickle.dump(model, ofile)
        return None
    
    @staticmethod
    def load(path):
        """
        Load a model from the specified location
        """
        with gzip.open(path, "r") as ifile:
            model = pickle.load(ifile)
        #
        new_model = Model2(torch.empty(model[0], model[11]), torch.empty(model[2], model[11]),
                           model[1], model[6],
                           model[3], model[4], model[5],
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
                           K_context=model[7]["K_context"])
        new_model.load_state_dict(model[10])
        #
        new_step = Step2(new_model.word_embedding.fast_text_embedding.weight.detach(), 
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
                         model[8], 
                         new_model)
        #
        return new_step
