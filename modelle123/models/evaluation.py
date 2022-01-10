#!/usr/bin/python3
"""
This file holds the code for a class collects methods for evaluation.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import numpy as np
import math


class Evaluation(object):
    """
    Class that holds methods for evaluation.
    """
    @staticmethod
    def binary_evaluation(data):
        """
        Performs a binary evaluatin. Data must be a list of tuples (gold, pred).
        gold: integer, 1 or -1
        pred: integer, 1 or -1
        """
        n = len(data)
        tp,fp,tn,fn = 0,0,0,0
        for i in range(n):
            y, y_hat = data[i]
            if (y==1) and (y_hat==1):
                tp += 1
            elif (y==1) and (y_hat==-1):
                fn += 1
            elif (y==-1) and (y_hat==1):
                fp += 1
            elif (y==-1) and (y_hat==-1):
                tn += 1
        p = tp / (tp+fp) if (tp+fp) > 0 else float("nan")
        r = tp / (tp+fn) if (tp+fn) > 0 else float("nan")
        f1 = (2*p*r) / (p+r) if (p>0) and (r>0) else float("nan")
        tnr = tn / (tn+fp) if (tn+fp) > 0 else float("nan")
        return {"p":p*100, "r":r*100, "f1":f1*100, "tpr":r*100, "tnr":100*tnr, "ba":(r+tnr)*100/2}
    
    @staticmethod
    def micro_multi_label_evaluation(data):
        """
        Performas a micro-averaged multi-label evaluation.
        Data must be a list of tuples (gold, pred)
            gold: set of labels
            pred: set of labels
        """
        n = len(data)
        corr, gold, pred = 0,0,0
        tp_indices = set()
        for i in range(n):
            y, y_hat = data[i]
            corr_pred = y.intersection(y_hat)
            corr += len(corr_pred)
            if len(corr_pred) > 0:
                tp_indices.add(i)
            gold += len(y)
            pred += len(y_hat)
        #
        p = corr / pred if pred>0 else float("nan")
        r = corr / gold if gold>0 else float("nan")
        f1 = 2*p*r / (p+r) if (p+r)>0 else float("nan")
        return {"p":100*p, "r":100*r, "f1":100*f1}, tp_indices
    
    @staticmethod
    def confusion_matrix_by_label(data, all_labels=None):
        """
        Creates a confusion-matrix for each label. Output is a dictionary holding labels as keys and dicts as values.
        Data must be a list of tuples (gold, pred)
            gold: set of labels
            pred: set of labels
        all_labels can specifie a list of labels that should be taken into account, others are ignored. Default is None.
        """
        confusions = dict()
        if all_labels is None:
            all_labels = set([l for y,_ in data for l in y]+[l for _,y_hat in data for l in y_hat])
        #
        indices_with_tps = set()
        # create confusions_matrix label wise
        for i in range(len(data)):
            y, y_hat = data[i]
            for actor in all_labels:
                if actor not in confusions:
                    confusions[actor] = {"tp":0, "fp":0, "tn":0, "fn":0}
                ##
                if (actor in y) and (actor in y_hat):
                    confusions[actor]["tp"] += 1
                    indices_with_tps.add(i)
                elif (actor in y) and (actor not in y_hat):
                    confusions[actor]["fn"] += 1
                elif (actor not in y) and (actor in y_hat):
                    confusions[actor]["fp"] += 1
                elif (actor not in y) and (actor not in y_hat):
                    confusions[actor]["tn"] += 1
        return confusions, indices_with_tps
    
    @staticmethod
    def micro_average_confusions(confusion_matrix, labels=None):
        """
        Micro averages a dictionary of confusion-matrices over all labels.
        confusion_matrix: Dictonary holding labels as keys and dicts as values.
        labels: can specify a list of labels to take into account for averaging, other are ignored then.
        Data must be a list of tuples (gold, pred)
            gold: set of labels
            pred: set of labels
        """
        if labels is None:
            labels = set(confusion_matrix.keys())
        #
        global_confusions = {"tp":0, "fp":0, "tn":0, "fn":0}
        for actor in labels:
            global_confusions["tp"] += confusion_matrix[actor]["tp"]
            global_confusions["fn"] += confusion_matrix[actor]["fn"]
            global_confusions["fp"] += confusion_matrix[actor]["fp"]
            global_confusions["tn"] += confusion_matrix[actor]["tn"]
        return global_confusions
    
    @staticmethod
    def confusions_to_metrics(confusion_matrix, tpr_tnr=True, p_r_f1=True):
        """
        Micro averages a dictionary of confusion-matrices over all labels.
        confusion_matrix: Dictonary holding fp, tp, fn, tn
        returns a dictionary holding various evaluation metrics.
        """
        metrics = dict()
        #
        if tpr_tnr is True:
            tpr, tnr = Evaluation.tp_tn_rate(confusion_matrix)
            metrics["tpr"] = tpr
            metrics["tnr"] = tnr
            metrics["ba"] = (tpr+tnr)/2
        if p_r_f1 is True:
            p, r, f1 = Evaluation.p_r_f1(confusion_matrix)
            metrics["p"] = p
            metrics["r"] = r
            metrics["f1"] = f1
        return metrics
    
    @staticmethod
    def tp_tn_rate(confusion_matrix):
        """
        Takes a confusion_matrix (dictionary) and returns a dictionary holding tp-rate and tn-rate.
        """
        denom = (confusion_matrix["tp"]+confusion_matrix["fn"]) 
        tpr = confusion_matrix["tp"] / denom if denom > 0 else float("nan")
        denom = (confusion_matrix["tn"]+confusion_matrix["fp"])
        tnr = confusion_matrix["tn"] / denom if denom > 0 else float("nan")
        return tpr*100, tnr*100
    
    @staticmethod
    def p_r_f1(confusion_matrix):
        """
        Takes a confusion_matrix (dictionary) and returns a dictionary holding precision, recall and f1.
        """
        denom = (confusion_matrix["tp"]+confusion_matrix["fp"]) 
        p = confusion_matrix["tp"] / denom if denom > 0 else float("nan")
        
        denom = (confusion_matrix["tp"]+confusion_matrix["fn"]) 
        r = confusion_matrix["tp"] / denom if denom > 0 else float("nan")
        
        denom = (p+r)
        f1 = (2*p*r) / denom if (denom > 0) and (math.isnan(denom) is False) else float("nan")
        return p*100, r*100, f1*100
