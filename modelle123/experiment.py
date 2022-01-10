#!/usr/bin/python3
"""
This file holds the code for training model-instances for a 5-fold-crossvalidation.
python3 exmperiment.py path1 path2 path3 path4
    path1: path to an output-directory
    path2: path to an actor-embedding-file
    path3: path to a fasttext-embedding-file
    path4: path to the debatenet-directory

All important code is in the main method. Other methods are helpers to organize the code.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""


import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pickle
import sys
import os


from models import IndexExtractor
from models import DataHandler
from models import Step1, Step2, Step3


def load_data(actor_embedding_path, fasttext_path, debatenet_path):
    #
    article_path = os.path.join(debatenet_path,"lre_data")
    claim_path = os.path.join(debatenet_path,"LRE.merged.removed_doublets.jsonl")
    entity_path = os.path.join(debatenet_path,"debatenetv2_added_entities.removed_doublets.jsonl")
    actor_mapping_path = os.path.join(debatenet_path,"entity_mapping_daten_raw.removed_doublets.jsonl")
    original_anno_path = os.path.join(debatenet_path,"original_annotation.jsonl")
    #
    fsttxt_matrix, fsttxt_map = DataHandler.load_embeddings(fasttext_path, create_unk=True, renorm=1.0) 
    actor_matrix, actor_map = DataHandler.load_embeddings(actor_embedding_path, renorm=1.0)
    #
    fsttxt_matrix = torch.FloatTensor(fsttxt_matrix)
    actor_matrix = torch.FloatTensor(actor_matrix)
    #
    data = DataHandler.load_mardy_data(article_path, claim_path, entity_path, actor_mapping_path, original_anno_path, actor_map)
    #
    return (fsttxt_matrix, fsttxt_map), (actor_matrix, actor_map), data


def load_default_params1():
    model_parameter = {"efeature_embedding_dim":20,
                       "char_embedding_dim":25,
                       "filters_per_channel":30,
                       "kernels":(3,),
                       "max_word_len":25,
                       "num_lstm_layers":2,
                       "ffnn_hidden_dim":150,
                       "num_ffnn_layers":2,
                       "embedding_dropout_p":0.5,
                       "hidden_dropout_p":0.3,
                       #
                       "use_context":True,
                       "context_size":10,
                       "K":5,
                       "update_weights":True,
                       #
                       "lambda":0.01}
    train_params = {"num_epochs":75,
                    "batch_size":16,
                    "learning_rate":1e-3,
                    "lr_reduction":0.5,
                    "patience":15,
                    "threshold":0.05}
    return model_parameter, train_params


def load_default_params2():
    model_parameter = {"efeature_embedding_dim":20,        # !
                       "cfeature_embedding_dim":40,        # !
                       "dfeature_embedding_dim":20,        # !
                       "char_embedding_dim":25,            # !
                       "filters_per_channel":30,           # !
                       "kernels":(3,),
                       "num_lstm_layers":2,
                       "ffnn_hidden_dim":150,
                       "num_ffnn_layers":2,
                       "embedding_dropout_p":0.5,
                       "hidden_dropout_p":0.3,
                       "max_word_len":25,
                       #
                       "update_claim_wise":True,
                       #
                       "loss_weights":[1,1],
                       #
                       "use_context":True,
                       "context_size":10,
                       "K_context":5,
                       #
                       "lambda":0.01,
                       "lambda_nil":0.01}
    train_params = {"num_epochs":75,
                    "batch_size":32,
                    "learning_rate":1e-3,
                    "lr_reduction":0.5,
                    "patience":15,
                    "threshold":0.05}
    return model_parameter, train_params


def load_default_params3():
    model_parameter = {"efeature_embedding_dim":20,        # !
                       "cfeature_embedding_dim":40,        # !
                       "dfeature_embedding_dim":20,        # !
                       "char_embedding_dim":25,            # !
                       "filters_per_channel":30,           # !
                       #
                       "kernels":(3,),
                       "num_lstm_layers":2,
                       "ffnn_hidden_dim":150,
                       "num_ffnn_layers":2,
                       "embedding_dropout_p":0.5,
                       "hidden_dropout_p":0.3,
                       "max_word_len":25,
                       #
                       "use_context":True,
                       "context_size":10,
                       "K_context":5,
                       #
                       "update_claim_wise":True,
                       #
                       "lambda":0.01}
    train_params = {"num_epochs":75,
                    "batch_size":32,
                    "learning_rate":1e-3,
                    "lr_reduction":0.5,
                    "patience":15,
                    "threshold":0.05}
    return model_parameter, train_params


def average_over_rounds(kfold_res):
    def dict_add(dst, src):
        for key, val in src.items():
            if key not in dst:
                dst[key] = []
            dst[key].append(val)
    def normalize_dict(dst, n):
        for k,v in dst.items():
            if isinstance(v, dict):
                normalize_dict(v, n)
            elif k in ["train_curves","eval_by_delta","attr_eval_by_delta", "nil_flag_eval_by_delta", "nil_attr_eval_by_delta", "examples"]:
                pass
            else:
                m = np.mean(v)
                s = np.std(v)
                dst[k] = {"mean":m, "std":s}
    global_results = {"model1":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "train_curves":[], "eval_by_delta":[]},
                      "model2":{"attr_test_results":{},"nil_attr_test_results":{},"nil_flag_test_results":{}, "attr_unseen_results":{}, "attr_freq_results":{},"nil_attr_freq_results":{}, "train_curves":[], "attr_eval_by_delta":[], "nil_flag_eval_by_delta":[], "nil_attr_eval_by_delta":[]},
                      "model3":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "train_curves":[], "eval_by_delta":[]},
                      "pipeline":{"model2":{"attr_test_results":{},"nil_attr_test_results":{},"nil_flag_test_results":{}, "attr_unseen_results":{}, "attr_freq_results":{},"nil_attr_freq_results":{}, "train_curves":[], "attr_eval_by_delta":[], "nil_flag_eval_by_delta":[], "nil_attr_eval_by_delta":[]},
                                  "model3":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "eval_by_delta":[]}}}
    for round_res in kfold_res:
        # Add results of model 1 to global results
        dict_add(global_results["model1"]["test_results"], round_res["model1"]["test_results"])
        dict_add(global_results["model1"]["unseen_results"], round_res["model1"]["unseen_results"])
        for bin_ in round_res["model1"]["freq_results"]:
            if bin_ not in global_results["model1"]["freq_results"]:
                global_results["model1"]["freq_results"][bin_] = {}
            dict_add(global_results["model1"]["freq_results"][bin_], round_res["model1"]["freq_results"][bin_])
        global_results["model1"]["train_curves"].append(round_res["model1"]["train_curves"])
        global_results["model1"]["eval_by_delta"].append(round_res["model1"]["eval_by_delta"])
        # Add results of model 2 to global results
        dict_add(global_results["model2"]["attr_test_results"], round_res["model2"]["attr_test_results"])
        dict_add(global_results["model2"]["nil_attr_test_results"], round_res["model2"]["nil_attr_test_results"])
        dict_add(global_results["model2"]["nil_flag_test_results"], round_res["model2"]["nil_flag_test_results"])
        dict_add(global_results["model2"]["attr_unseen_results"], round_res["model2"]["attr_unseen_results"])
        for bin_ in round_res["model2"]["attr_freq_results"]:
            if bin_ not in global_results["model2"]["attr_freq_results"]:
                global_results["model2"]["attr_freq_results"][bin_] = {}
            dict_add(global_results["model2"]["attr_freq_results"][bin_], round_res["model2"]["attr_freq_results"][bin_])
        for bin_ in round_res["model2"]["nil_attr_freq_results"]:
            if bin_ not in global_results["model2"]["nil_attr_freq_results"]:
                global_results["model2"]["nil_attr_freq_results"][bin_] = {}
            dict_add(global_results["model2"]["nil_attr_freq_results"][bin_], round_res["model2"]["nil_attr_freq_results"][bin_])
        global_results["model2"]["train_curves"].append(round_res["model2"]["train_curves"])
        global_results["model2"]["attr_eval_by_delta"].append(round_res["model2"]["attr_eval_by_delta"])
        global_results["model2"]["nil_attr_eval_by_delta"].append(round_res["model2"]["nil_attr_eval_by_delta"])
        global_results["model2"]["nil_flag_eval_by_delta"].append(round_res["model2"]["nil_flag_eval_by_delta"])
        # Add results of model 3 to global results
        dict_add(global_results["model3"]["test_results"], round_res["model3"]["test_results"])
        dict_add(global_results["model3"]["unseen_results"], round_res["model3"]["unseen_results"])
        for bin_ in round_res["model3"]["freq_results"]:
            if bin_ not in global_results["model3"]["freq_results"]:
                global_results["model3"]["freq_results"][bin_] = {}
            dict_add(global_results["model3"]["freq_results"][bin_], round_res["model3"]["freq_results"][bin_])
        global_results["model3"]["train_curves"].append(round_res["model3"]["train_curves"])
        global_results["model3"]["eval_by_delta"].append(round_res["model3"]["eval_by_delta"])
        
        # Add results ofpipeline to global results
        dict_add(global_results["pipeline"]["model2"]["attr_test_results"], round_res["pipeline"]["model2"]["attr_test_results"])
        dict_add(global_results["pipeline"]["model2"]["nil_attr_test_results"], round_res["pipeline"]["model2"]["nil_attr_test_results"])
        dict_add(global_results["pipeline"]["model2"]["nil_flag_test_results"], round_res["pipeline"]["model2"]["nil_flag_test_results"])
        dict_add(global_results["pipeline"]["model2"]["attr_unseen_results"], round_res["pipeline"]["model2"]["attr_unseen_results"])
        for bin_ in round_res["pipeline"]["model2"]["attr_freq_results"]:
            if bin_ not in global_results["pipeline"]["model2"]["attr_freq_results"]:
                global_results["pipeline"]["model2"]["attr_freq_results"][bin_] = {}
            dict_add(global_results["pipeline"]["model2"]["attr_freq_results"][bin_], round_res["pipeline"]["model2"]["attr_freq_results"][bin_])
        for bin_ in round_res["pipeline"]["model2"]["nil_attr_freq_results"]:
            if bin_ not in global_results["pipeline"]["model2"]["nil_attr_freq_results"]:
                global_results["pipeline"]["model2"]["nil_attr_freq_results"][bin_] = {}
            dict_add(global_results["pipeline"]["model2"]["nil_attr_freq_results"][bin_], round_res["pipeline"]["model2"]["nil_attr_freq_results"][bin_])
        global_results["pipeline"]["model2"]["attr_eval_by_delta"].append(round_res["pipeline"]["model2"]["attr_eval_by_delta"])
        global_results["pipeline"]["model2"]["nil_attr_eval_by_delta"].append(round_res["pipeline"]["model2"]["nil_attr_eval_by_delta"])
        global_results["pipeline"]["model2"]["nil_flag_eval_by_delta"].append(round_res["pipeline"]["model2"]["nil_flag_eval_by_delta"])
        #
        dict_add(global_results["pipeline"]["model3"]["test_results"], round_res["pipeline"]["model3"]["test_results"])
        dict_add(global_results["pipeline"]["model3"]["unseen_results"], round_res["pipeline"]["model3"]["unseen_results"])
        for bin_ in round_res["pipeline"]["model3"]["freq_results"]:
            if bin_ not in global_results["pipeline"]["model3"]["freq_results"]:
                global_results["pipeline"]["model3"]["freq_results"][bin_] = {}
            dict_add(global_results["pipeline"]["model3"]["freq_results"][bin_], round_res["pipeline"]["model3"]["freq_results"][bin_])
        global_results["pipeline"]["model3"]["eval_by_delta"].append(round_res["pipeline"]["model3"]["eval_by_delta"])
        
    # Calaculate mean and std for merged data
    normalize_dict(global_results, len(kfold_res))
    return global_results


def main(output_dir_path, actor_embedding_path, fasttext_path, debatenet_path):
    # Load all data needed
    (fsttxt_matrix, fsttxt_map), (actor_matrix, actor_map), data = load_data(actor_embedding_path, fasttext_path, debatenet_path)
    # Prepare a dictionary holding actor counts (from full data)
    # Prepare a set of all known actors (actors with wikidata-id)
    all_categories = set([cc for d in data for c in d.stacked_claims for cc in c.categories])
    category_map = dict([(c,n) for n,c in enumerate(sorted(all_categories))])
    print("Num_Categories", len(all_categories))
    # Load the default parameters for the models
    model_parameters1, train_parameters1 = load_default_params1()
    model_parameters2, train_parameters2 = load_default_params2()
    model_parameters3, train_parameters3 = load_default_params3()
    # Do a 5-fold-crossvalidation
    kfold_results = []
    folds =  DataHandler.stratified_kfold_iterator(data, dev_ratio=1/4, num_folds=5)
    fold_counter = 0
    for train, dev, test in folds:
        fold_counter += 1
        print("Split: {} / {} / {}".format(len(train), len(dev), len(test)))
        #
        DataHandler.write_documents(train, os.path.join(output_dir_path, "fold{}_train.bin".format(fold_counter)))
        DataHandler.write_documents(dev, os.path.join(output_dir_path, "fold{}_dev.bin".format(fold_counter)))
        DataHandler.write_documents(test, os.path.join(output_dir_path, "fold{}_test.bin".format(fold_counter)))
        # Prepare data
        extractor = IndexExtractor(word_map=fsttxt_map,
                                   actor_map=actor_map,
                                   category_map=category_map)
        pad_idx = extractor.char_mapper.lookup("[PAD]")
        extractor.extract_all_gold(train)
        extractor.freeze()
        extractor.save(os.path.join(output_dir_path, "fold{}_extractor.bin".format(fold_counter)))
        extractor.extract_all_gold(dev)
        extractor.extract_all_gold(test)
        # Debug count ratio of known words
        tot = 0
        unk = 0
        for d in data:
            for t in d.tokens:
                if t.i_fasttext == 0:
                    unk += 1
                tot += 1
        print("Ratio of unknown words: {} / {}   (= {}%)".format(unk, tot, unk*100/tot))
        # Train models
        model1 = Step1(fsttxt_matrix, actor_matrix,
                       word_vocab_size=len(extractor.word_map()),
                       char_vocab_size=len(extractor.char_map()),
                       actor_vocab_size=len(extractor.actor_map()),
                       feature_vocab_size=len(extractor.entity_map()),
                       char_pad_index=pad_idx,
                       model_params=model_parameters1)
        train_book1 = model1.train(train, dev,
                                   num_epochs=train_parameters1["num_epochs"],
                                   batch_size=train_parameters1["batch_size"],
                                   learning_rate=train_parameters1["learning_rate"],
                                   lr_reduction=train_parameters1["lr_reduction"],
                                   patience=train_parameters1["patience"],
                                   threshold=train_parameters1["threshold"],
                                   cuda=True)
        print("\n\n")
        model2 = Step2(fsttxt_matrix, actor_matrix,
                       word_vocab_size=len(extractor.word_map()),
                       char_vocab_size=len(extractor.char_map()),
                       actor_vocab_size=len(extractor.actor_map()),
                       entity_feature_vocab_size=len(extractor.entity_map()),
                       claim_feature_vocab_size=len(extractor.category_map()),
                       dist_feature_vocab_size=len(extractor.distance_map()),
                       char_pad_index=pad_idx,
                       model_params=model_parameters2)
        train_book2 = model2.train(train, dev,
                                   num_epochs=train_parameters2["num_epochs"],
                                   batch_size=train_parameters2["batch_size"],
                                   learning_rate=train_parameters2["learning_rate"],
                                   lr_reduction=train_parameters2["lr_reduction"],
                                   patience=train_parameters2["patience"],
                                   threshold=train_parameters1["threshold"],
                                   cuda=True)
        print("\n\n")
        model3 = Step3(fsttxt_matrix, actor_matrix,
                       word_vocab_size=len(extractor.word_map()),
                       char_vocab_size=len(extractor.char_map()),
                       actor_vocab_size=len(extractor.actor_map()),
                       entity_feature_vocab_size=len(extractor.entity_map()),
                       claim_category_vocab_size=len(extractor.category_map()),
                       distance_feature_vocab_size=len(extractor.distance_map()),
                       char_pad_idx=pad_idx,
                       model_params=model_parameters3)
        train_book3 = model3.train(train, dev,
                                   num_epochs=train_parameters3["num_epochs"],
                                   batch_size=train_parameters3["batch_size"],
                                   learning_rate=train_parameters3["learning_rate"],
                                   lr_reduction=train_parameters3["lr_reduction"],
                                   patience=train_parameters3["patience"],
                                   threshold=train_parameters1["threshold"],
                                   cuda=True)
        print("\n\n")
        #
        model1.store(os.path.join(output_dir_path, "fold{}_model1.trained.bin".format(fold_counter)))
        model2.store(os.path.join(output_dir_path, "fold{}_model2.trained.bin".format(fold_counter)))
        model3.store(os.path.join(output_dir_path, "fold{}_model3.trained.bin".format(fold_counter)))
    return None


if __name__=="__main__":
    if len(sys.argv) != 5:
        print("Please specify paths to an output directory, the actor embeddings, fasttext embeddings, path to debatenet data.")
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    exit()

