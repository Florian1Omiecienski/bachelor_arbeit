#!/usr/bin/python3
"""
This file holds the code for evaluation models that were created during a 5-fold-crossvalidation.
Use this program after experiment.py
python3 evaluate.py path1 path2 path3
    path1: path to an output-directory
    path2: path to an actor-embedding-file
    path3: path to the debatenet-directory

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


from models import Evaluation
from models import IndexExtractor
from models import DataHandler
from models import Step1, Step2, Step3


def _bin(val, bins=(1,2,3,4,5,6,7,8,16,32,64)):
    for i in range(len(bins)-1):
        lower = bins[i]
        upper = bins[i+1]
        if (lower <= val) and (val < upper):
            return lower
    return bins[-1]


def load_data(actor_embedding_path, debatenet_path):
    #
    article_path = os.path.join(debatenet_path,"lre_data")
    claim_path = os.path.join(debatenet_path,"LRE.merged.removed_doublets.jsonl")
    entity_path = os.path.join(debatenet_path,"debatenetv2_added_entities.removed_doublets.jsonl")
    actor_mapping_path = os.path.join(debatenet_path,"entity_mapping_daten_raw.removed_doublets.jsonl")
    original_anno_path = os.path.join(debatenet_path,"original_annotation.jsonl")
    #
    _, actor_map = DataHandler.load_embeddings(actor_embedding_path, renorm=1.0)
    #
    data = DataHandler.load_mardy_data(article_path, claim_path, entity_path, actor_mapping_path, original_anno_path, actor_map)
    #
    return data


def get_actor_doc_freqs(data):
    actor_counts = {}
    for doc in data:
        for a in doc.actors:
            if a.is_nil is False:
                if a.wikidata_id not in actor_counts:
                    actor_counts[a.wikidata_id] = 0
                actor_counts[a.wikidata_id] += 1
    return actor_counts


def evaluate_confusions(confusions, label_set):
    micro_confusions = Evaluation.micro_average_confusions(confusions, label_set)
    metrics = Evaluation.confusions_to_metrics(micro_confusions)
    import math
    if math.isnan(metrics["p"]):
        print("NAN WARNING")
    return metrics


def evaluate_confusions_by_count(confusions, label_counts):
    test_actors_by_freq = {}
    # sort actors by frequency-bins
    for a in confusions.keys():
        if a not in label_counts:     # can only happen if more embeddings then actors
            continue
        ac = _bin(label_counts[a])
        if ac not in test_actors_by_freq:
            test_actors_by_freq[ac] = []
        test_actors_by_freq[ac].append(a)
    # evaluate for each bin
    # over all associated actors
    micro_by_freq = {}
    for freq, actors in test_actors_by_freq.items():
        micro_by_freq[freq] = evaluate_confusions(confusions, actors)
    return micro_by_freq


def create_actor_doc_confusions(test_data):
    predictions = []
    for doc in test_data:
        y = set([a.wikidata_id for a in doc.actors if a.is_nil is False])
        y_hat = set([a.p_wikidata_id for a in doc.p_actors])
        predictions.append((y, y_hat))
    conf_matrix, tp_indices = Evaluation.confusion_matrix_by_label(predictions)
    # Find docs with predicted tps
    tp_docs = [test_data[i] for i in set(tp_indices)]
    return conf_matrix, tp_docs


def create_attribution_confusions(test_data):
    predictions = []
    claim2doc_map = []
    for doc in test_data:
        for i in range(len(doc.stacked_claims)):
            y = set([a.wikidata_id for a in doc.stacked_claims[i].actors if a.is_nil is False])
            y_hat = set([a.p_wikidata_id for a in doc.p_stacked_claims[i].p_actors if a.is_nil is False])
            predictions.append((y, y_hat))
            claim2doc_map.append(doc)
    conf_matrix, tp_indices = Evaluation.confusion_matrix_by_label(predictions)
    #
    tp_docs = [claim2doc_map[i] for i in tp_indices]
    return conf_matrix, tp_docs


def create_nil_confusions(test_data):
    predictions = []
    claim2doc_map = []
    for doc in test_data:
        for i in range(len(doc.stacked_claims)):
            y = set([a.normal_name for a in doc.stacked_claims[i].actors if a.is_nil is True])
            y_hat = set([a.p_normal_name for a in doc.p_stacked_claims[i].p_actors if a.is_nil is True])
            predictions.append((y, y_hat))
            claim2doc_map.append(doc)
    conf_matrix,tp_indices = Evaluation.confusion_matrix_by_label(predictions)
    #
    tp_docs = [claim2doc_map[i] for i in tp_indices]
    return conf_matrix, tp_docs


def evaluate_nil_flags_binary(test_data):
    data2eval = []
    for doc in test_data:
        for gclaim, pclaim in zip(doc.stacked_claims, doc.p_stacked_claims):
            gold_flag = any([a.is_nil for a in gclaim.actors])
            gold_flag = 1 if gold_flag is True else -1
            pred_flag = any([a.is_nil for a in pclaim.p_actors])
            pred_flag = 1 if pred_flag is True else -1
            data2eval.append((gold_flag, pred_flag))
    res = Evaluation.binary_evaluation(data2eval)
    return res 


def evaluate_polarities(test_data, all_actors):
    predictions = []
    for doc in test_data:
        # read gold relations 
        y = dict()
        for i in range(len(doc.claims)):
            claim =  doc.claims[i]
            if claim.actor.is_nil is False:
                for cat in claim.categories:
                    y[(claim.actor.wikidata_id, cat)] = claim.polarity
        # read predicted relations
        y_hat = dict()
        for i in range(len(doc.p_claims)):
            claim =  doc.p_claims[i]
            if claim.p_actor.is_nil is False:
                for cat in claim.p_categories:
                    y_hat[(claim.p_actor.p_wikidata_id, cat)] = claim.p_polarity
        for k in set(y.keys()).union(set(y_hat.keys())):
            if k[0] in all_actors:
                y_flag = y[k]
                y_hat_flag = y_hat[k]
                predictions.append((int(y_flag), int(y_hat_flag)))
    # create confusion matrix
    return Evaluation.binary_evaluation(predictions)


def evaluate_relations(test_data, test_actors):
    data2eval = []
    index2doc = []
    for i in range(len(test_data)):
        doc = test_data[i]
        # create gold relations 
        for j in range(len(doc.stacked_claims)):
            y = set()
            for k in range(len(doc.stacked_claims[j].unstacked_claims)):
                claim =  doc.stacked_claims[j].unstacked_claims[k]
                if (claim.actor.is_nil is False) and (claim.polarity == "1"):  # TODO
                    if (test_actors is not None) and (claim.actor.wikidata_id not in test_actors):
                        continue
                    for cat in claim.categories:
                        y.add((claim.actor.wikidata_id, claim.polarity, cat))
            # create predicted relations
            y_hat = set()
            for k in range(len(doc.p_stacked_claims[j].p_unstacked_claims)):
                claim =  doc.p_stacked_claims[j].p_unstacked_claims[k]
                if (claim.p_actor.is_nil is False) and (claim.p_polarity == "1"):  # TODO
                    if (test_actors is not None) and (claim.p_actor.p_wikidata_id not in test_actors):
                        continue
                    for cat in claim.p_categories:
                        y_hat.add((claim.p_actor.p_wikidata_id, claim.p_polarity, cat))
            data2eval.append((y, y_hat))
            index2doc.append(doc)
    res, tp_indices = Evaluation.micro_multi_label_evaluation(data2eval)
    tp_docs = [index2doc[i] for i in tp_indices]
    return res, tp_docs


def evaluate_polarities_by_count(test_data, label_counts):
    test_actors_by_freq = {}
    for a in label_counts:
        ac = _bin(label_counts[a])
        if ac not in test_actors_by_freq:
            test_actors_by_freq[ac] = []
        test_actors_by_freq[ac].append(a)
    micro_by_freq = {}
    for freq, actors in test_actors_by_freq.items():
        res,_ = evaluate_relations(test_data, actors)
        micro_by_freq[freq] = res
    return micro_by_freq


def evaluate_model1(test, unseen_actors, actor_doc_freq):
    # Create confusion matrix per actor
    confusions, ex_docs = create_actor_doc_confusions(test)
    # Micro-Average confusions
    test_results = evaluate_confusions(confusions, None)
    unseen_results = evaluate_confusions(confusions, unseen_actors)
    # Evaluate by actor-doc-freq
    freq_results = evaluate_confusions_by_count(confusions, actor_doc_freq)
    # 
    results = {"test_results":test_results,
               "unseen_results":unseen_results,
               "freq_results":freq_results,
               "examples":ex_docs}
    #
    return results


def evaluate_model2(test, unseen_actors, actor_doc_freq):
    # Create confusion matrix per actor
    attr_confusions, ex_docs1 = create_attribution_confusions(test)
    nil_confusions, ex_docs2 = create_nil_confusions(test)
    ex_docs = set(ex_docs1+ex_docs2)
    # Micro-Average confusions
    attr_test_results = evaluate_confusions(attr_confusions, None)
    attr_unseen_results = evaluate_confusions(attr_confusions, unseen_actors)
    nil_attr_test_results = evaluate_confusions(nil_confusions, None)
    #
    nil_flag_test_results = evaluate_nil_flags_binary(test)
    # Evaluate by actor-doc-freq
    attr_freq_results = evaluate_confusions_by_count(attr_confusions, actor_doc_freq)
    nil_attr_freq_results = evaluate_confusions_by_count(nil_confusions, actor_doc_freq)
    # 
    results = {"attr_test_results":attr_test_results,
               "attr_unseen_results":attr_unseen_results,
               "nil_attr_test_results":nil_attr_test_results,
               "nil_flag_test_results":nil_flag_test_results,
               "attr_freq_results":attr_freq_results,
               "nil_attr_freq_results":nil_attr_freq_results,
               "examples":ex_docs,
              }
    #
    return results


def evaluate_model3(test, unseen_actors, actor_doc_freq):
    test_results, ex_docs = evaluate_relations(test, None)
    unseen_results,_ = evaluate_relations(test, unseen_actors)
    # Evaluate by actor-doc-freq
    freq_results = evaluate_polarities_by_count(test, actor_doc_freq)
    # 
    results = {"test_results":test_results,
               "unseen_results":unseen_results,
               "freq_results":freq_results,
               "examples":ex_docs}
    #
    return results


def evaluate_pipeline(test, unseen_actors, actor_doc_freq):
    model2_res = evaluate_model2(test, unseen_actors, actor_doc_freq)
    model3_res = evaluate_model3(test, unseen_actors, actor_doc_freq)
    return {"model2":model2_res,
            "model3":model3_res}


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
    global_results = {"model1":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "eval_by_delta":[]},
                      "model2":{"attr_test_results":{},"nil_attr_test_results":{},"nil_flag_test_results":{}, "attr_unseen_results":{}, "attr_freq_results":{},"nil_attr_freq_results":{}, "attr_eval_by_delta":[], "nil_flag_eval_by_delta":[], "nil_attr_eval_by_delta":[]},
                      "model3":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "eval_by_delta":[]},
                      "pipeline":{"model2":{"attr_test_results":{},"nil_attr_test_results":{},"nil_flag_test_results":{}, "attr_unseen_results":{}, "attr_freq_results":{},"nil_attr_freq_results":{}, "attr_eval_by_delta":[], "nil_flag_eval_by_delta":[], "nil_attr_eval_by_delta":[]},
                                  "model3":{"test_results":{}, "unseen_results":{}, "freq_results":{}, "eval_by_delta":[]}}}
    for round_res in kfold_res:
        # Add results of model 1 to global results
        dict_add(global_results["model1"]["test_results"], round_res["model1"]["test_results"])
        dict_add(global_results["model1"]["unseen_results"], round_res["model1"]["unseen_results"])
        for bin_ in round_res["model1"]["freq_results"]:
            if bin_ not in global_results["model1"]["freq_results"]:
                global_results["model1"]["freq_results"][bin_] = {}
            dict_add(global_results["model1"]["freq_results"][bin_], round_res["model1"]["freq_results"][bin_])
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


def main(output_dir_path, actor_embedding_path, debatenet_path, recall_ratio=0.9):
    # Load full data
    print("loading data ...")
    data = load_data(actor_embedding_path, debatenet_path)
    # Prepare a dictionary holding actor counts (from full data)
    actor_doc_freqs = get_actor_doc_freqs(data)
    #
    kfold_results = []
    # Do a 5-fold-crossvalidation
    for fold in range(1,6):
        print("Fold-{}".format(fold))
        # Load data split
        train = DataHandler.load_documents(os.path.join(output_dir_path, "fold{}_train.bin".format(fold)))
        dev = DataHandler.load_documents(os.path.join(output_dir_path, "fold{}_dev.bin".format(fold)))
        test = DataHandler.load_documents(os.path.join(output_dir_path, "fold{}_test.bin".format(fold)))
        # Load index extractor
        extractor = IndexExtractor.load(os.path.join(output_dir_path, "fold{}_extractor.bin".format(fold)))
        pad_idx = extractor.char_mapper.lookup("[PAD]")
        extractor.extract_all_gold(train)
        extractor.extract_all_gold(dev)
        extractor.extract_all_gold(test)
        # Load models
        model1 = Step1.load(os.path.join(output_dir_path, "fold{}_model1.trained.bin".format(fold)))
        model2 = Step2.load(os.path.join(output_dir_path, "fold{}_model2.trained.bin".format(fold)))
        model3 = Step3.load(os.path.join(output_dir_path, "fold{}_model3.trained.bin".format(fold)))
        # Prepare set of unseen actors to evaluate over
        train_actors = set([a.wikidata_id for d in train for a in d.actors if a.is_nil is False])
        dev_actors   = set([a.wikidata_id for d in dev for a in d.actors if a.is_nil is False])
        test_actors  = set([a.wikidata_id for d in test for a in d.actors if a.is_nil is False])
        unseen_actors = test_actors.difference(train_actors.union(dev_actors))
        print("\tUNSEEN:", len(unseen_actors))
        ### Evaluate models seperatly
        # Set prediction threshold for models to max_recall (90%)
        delta1_range = model1.estimate_delta_range(dev, recall_ratio=recall_ratio,cuda=True)
        delta2_range, delta3_range = model2.estimate_delta_ranges(dev, recall_ratio=recall_ratio, cuda=True)
        delta4_range = model3.estimate_delta_range(dev, recall_ratio=recall_ratio, cuda=True)
        model1.delta = delta1_range[0]
        model2.delta = delta2_range[0]
        model2.delta_nil = delta3_range[0]
        model3.delta = delta4_range[0]
        # Estimate prediction thresholds for max_recall and max_precision for later pr-curves
        delta1_range = model1.estimate_delta_range(dev, recall_ratio=1,cuda=True)
        delta2_range, delta3_range = model2.estimate_delta_ranges(dev, recall_ratio=1, cuda=True)
        delta4_range = model3.estimate_delta_range(dev, recall_ratio=1, cuda=True)
        # Predict test data (each model takes gold data as input)
        print("\tPredicting ...")
        model1.predict(test, cuda=True)
        model2.predict(test, cuda=True)
        model3.predict(test, cuda=True)
        extractor.inverse_extract_all_predictions(test)
        # Evaluate trained models seperatly
        res1 = evaluate_model1(test, unseen_actors, actor_doc_freqs)
        res1["eval_by_delta"] = model1.evaluate_delta_range(delta1_range, test, num_steps=50, cuda=True)
        #
        res2 = evaluate_model2(test, unseen_actors, actor_doc_freqs)
        res2["attr_eval_by_delta"] = model2.evaluate_delta_range(delta2_range, test, num_steps=50, cuda=True)
        nil_flag_by_delta, nil_by_delta = model2.evaluate_delta_nil_range(delta3_range, test, num_steps=50, cuda=True)
        res2["nil_flag_eval_by_delta"] = nil_flag_by_delta
        res2["nil_attr_eval_by_delta"] = nil_by_delta
        #
        res3 = evaluate_model3(test, unseen_actors, actor_doc_freqs)
        res3["eval_by_delta"] = model3.evaluate_delta_range(delta4_range, test, num_steps=50, cuda=True)
        
        ### Evaluate trained models as pipeline
        # Predict the test data using the estimated deltas
        print("\tPredicting ...")
        model1.predict(test, cuda=True)
        model2.predict(test, input_predictions=True, cuda=True)
        model3.predict(test, input_predictions=True, cuda=True)
        extractor.inverse_extract_all_predictions(test)
        # Evaluate predicted data
        res_pipe = evaluate_pipeline(test, unseen_actors, actor_doc_freqs)
        res_pipe["model2"]["attr_eval_by_delta"] = model2.evaluate_delta_range(delta2_range, test, input_predictions=True, num_steps=50, cuda=True)
        nil_flag_by_delta, nil_by_delta = model2.evaluate_delta_nil_range(delta3_range, test, input_predictions=True, num_steps=50, cuda=True)
        res_pipe["model2"]["nil_flag_eval_by_delta"] = nil_flag_by_delta
        res_pipe["model2"]["nil_attr_eval_by_delta"] = nil_by_delta
        res_pipe["model3"]["eval_by_delta"] = model3.evaluate_delta_range(delta4_range, test, input_predictions=True, num_steps=50, cuda=True)
        # Store results for this round
        round_results = {"model1":res1,
                         "model2":res2,
                         "model3":res3,
                         "pipeline":res_pipe}
        kfold_results.append(round_results)
    # After all round are done
    average_result = average_over_rounds(kfold_results)
    average_result["round_results"] = kfold_results
    # Store resuls in file
    with open(os.path.join(output_dir_path, "evaluation_results.bin"), "bw") as ofile:
        pickle.dump(average_result, ofile)
    # end
    return None


if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Please specify paths to an output directory, the actor embeddings, path to debatenet data.")
        exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    exit()
