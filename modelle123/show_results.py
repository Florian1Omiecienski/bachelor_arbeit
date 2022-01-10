#!/usr/bin/python3
"""
This file holds the code showing the results of the evaluate.py.
Shows the results in a humanreadable form in the console.
De-comment lines in the main-method to add graphics or qualitative-examples.

python3 show_results.py path1
    path1: path to an result-file created with evaluate.py

All important code is in the main method. Other methods are helpers to organize the code.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""

import matplotlib.pyplot as plt
from math import isnan
import numpy as np
import pickle
import sys


def plot_average_metrics(metric_test, metric_unseen):
    #
    means = [metric_test["p"]["mean"],
             metric_test["r"]["mean"],
             metric_test["f1"]["mean"]]
    stds = [metric_test["p"]["std"],
            metric_test["r"]["std"],
            metric_test["f1"]["std"]]
    places = [2/3,1,4/3]
    plt.bar(places, means, yerr=stds, width=1/3, color=["b","g","r"])
    #
    means = [metric_unseen["p"]["mean"],
             metric_unseen["r"]["mean"],
             metric_unseen["f1"]["mean"]]
    stds = [metric_unseen["p"]["std"],
            metric_unseen["r"]["std"],
            metric_unseen["f1"]["std"]]
    places = [2+(2/3),2+1,2+(4/3)]
    plt.bar(places, means, yerr=stds, width=1/3, color=["b","g","r"])
    #
    plt.title("Micro averaged metrices over full test data (left) \nand over zero-shot actors only (right)")
    plt.xticks([2/3,1,4/3,2+(2/3),2+1,2+(4/3)], ["p", "r", "f1","p", "r", "f1"])
    plt.show()


def plot_metrics_by_freq(micro_by_freq):
    # Plot
    counts, means, stds = zip(*[(c, v["f1"]["mean"], v["f1"]["std"]) for c,v in sorted(micro_by_freq.items(), key=lambda x:x[0])])
    N = len(counts)
    plt.title("Micro averaged evaluation results for different actor-counts")
    plt.xlabel("binned actor frequency")
    plt.ylabel("F1 (%)")
    plt.xticks(range(N), labels=[str(c) for c in counts])
    plt.bar(range(N), means,yerr=stds)
    plt.show()


def plot_prc(delta_range_data, no_skill=50):
    N = len(delta_range_data)
    for i in range(N):
        metrics_by_delta = delta_range_data[i]
        delta_range = [m["delta"] for m in metrics_by_delta]
        precision_range = [m["p"] for m in metrics_by_delta]
        recall_range = [m["r"] for m in metrics_by_delta]
        plt.plot(recall_range, precision_range)
        plt.xlabel("Recall (%)")
        plt.ylabel("Precison (%)")
        ax = plt.gca()
        ax.annotate("{:.2f}".format(delta_range[0]), (recall_range[0], precision_range[0]))
        ax.annotate("{:.2f}".format(delta_range[-1]), (recall_range[-1], precision_range[-1]))#
    plt.plot(range(100), [no_skill,]*100, linestyle="--", c="r", label="no-skill-threshold")
    plt.title("Precision-Recall-Curves over diffrent values of delta")
    plt.legend()
    plt.show()


def plot_roc(delta_range_data):
    N = len(delta_range_data)
    for i in range(N):
        metrics_by_delta = delta_range_data[i]
        delta_range = [m["delta"] for m in metrics_by_delta]
        tnr_range = [m["tnr"] for m in metrics_by_delta]
        tpr_range = [m["tpr"] for m in metrics_by_delta]
        plt.plot(tnr_range, tpr_range)
        plt.xlabel("TNR (%)")
        plt.ylabel("TPR (%)")
        ax = plt.gca()
        ax.annotate("{:.2f}".format(delta_range[0]), (tnr_range[0], tpr_range[0]))
        ax.annotate("{:.2f}".format(delta_range[-1]), (tnr_range[-1], tpr_range[-1]))#
    plt.plot(range(100), range(100,0,-1), linestyle="--", c="r", label="no-skill-threshold")
    plt.title("RO-Curves over diffrent values of delta")
    plt.legend()
    plt.show()


def plot_train_curves(train_curves, dev_curves):
    for i in range(len(train_curves)):
        #
        pt_curve = [epoch["p"] for epoch in train_curves[i]]
        rt_curve = [epoch["r"] for epoch in train_curves[i]]
        #
        pd_curve = [epoch["p"] for epoch in dev_curves[i]]
        rd_curve = [epoch["r"] for epoch in dev_curves[i]]
        #
        x_range = list(range(len(pt_curve)))
        #
        plt.title("Evaluation during training")
        plt.subplot(121)
        plt.title("Train-Set")
        plt.plot(x_range, pt_curve, c="b", label="P")
        plt.plot(x_range, rt_curve, c="orange", label="R")
        plt.xlabel("epoch")
        plt.subplot(122)
        plt.title("Dev-Set")
        plt.plot(x_range, pd_curve, c="b", label="P")
        plt.plot(x_range, rd_curve, c="orange", label="R")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()


def model1_example(docs):
    examples = dict()
    for doc in docs:
        doc_id = doc.doc_id
        doctext = ""
        #
        for i in range(len(doc.tokens)):
            text = doc.tokens[i].text
            for a in doc.p_actors:
                aspan = a.p_spans[0]
                if (i==aspan[0]):
                    doctext += "<Actor-{:s}> ".format(a.p_wikidata_id)
                if (i==aspan[1]):
                    doctext += "</Actor-{:s}> ".format(a.p_wikidata_id)
            doctext += "{:s} ".format(text)
        #
        examples[doc_id] = doctext
    return examples


def model2_example(docs):
    examples = dict()
    for doc in docs:
        doc_id = doc.doc_id
        doctext = ""
        #
        for i in range(len(doc.tokens)):
            text = doc.tokens[i].text
            claim_annotated = False
            for c in doc.p_stacked_claims:
                cspan = c.span
                if (i==cspan[1]):
                    doctext += "</Claim-{:s}> ".format(c.anno_id)
                if (i==cspan[0]):
                    doctext += "<Claim-{:s}> ".format(c.anno_id)
                for a in c.p_actors:
                    if a.is_nil is True:
                        type_ = "NIL"
                        
                    else:
                        type_ = "Actor-{:s}".format(a.p_wikidata_id)
                    aspan = a.p_spans[0]
                    if (i==aspan[0]):
                        doctext += "<{:s}-{:s}> ".format(type_, c.anno_id)
                    if (i==aspan[1]):
                        doctext += "</{:s}-{:s}> ".format(type_, c.anno_id)
            doctext += "{:s} ".format(text)
        #
        examples[doc_id] = doctext
    return examples


def model3_example(docs):
    examples = dict()
    for doc in docs:
        doc_id = doc.id
        #
        text = None
        examples[doc_id] = text
    return examples


def main(path):
    # Load evaluation file
    glob_res = pickle.load(open(path, "rb"))
    """  Kommentar entfernen um qualitative Beispiel des ersten Modells zu zeigen
    print("-"*10+" Qualitative Beispiele Modell 1 "+"-"*10)
    print()
    for doc in glob_res["round_results"][3]["model1"]["examples"]:
        print(">>> Doc-ID: ", doc.doc_id)
        print(model1_example([doc,]))
        print(">>> Gold: ", [a.wikidata_id for a in doc.actors if a.is_nil])
        input()
    """

    """  Kommentar entfernen um qualitative Beispiel des zweiten Modells zu zeigen
    print("-"*10+" Qualitative Beispiele Modell 2 "+"-"*10)
    print()
    for doc in glob_res["round_results"][3]["model2"]["examples"]:
        if len(doc.stacked_claims) != 2:
            continue
        print("----- Gold-Claims")
        print(doc.stacked_claims[0])
        print(doc.stacked_claims[1])
        print("----- Pred-Claims")
        print(doc.p_stacked_claims[0])
        print(doc.p_stacked_claims[1])
        print("----- Gold-Candidates:")
        print([a.normal_name for a in doc.actors])
        print("----- Pred-Candidates:")
        print([a.p_wikidata_id for a in doc.p_actors])
        print("-----")
        print(">>> Doc-ID: ", doc.doc_id)
        print(model2_example([doc,]))
        print(">>> Gold: ", [a.wikidata_id for a in doc.actors if a.is_nil is False])
        input()
    """

    print("-"*10+" Ergebnisse der Modell einzeln "+"-"*10)
    print()
    print("Modell 1")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model1"]["test_results"]["p"]["mean"],
                                                         glob_res["model1"]["test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model1"]["test_results"]["r"]["mean"],
                                                         glob_res["model1"]["test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model1"]["test_results"]["f1"]["mean"],
                                                         glob_res["model1"]["test_results"]["f1"]["std"],))
    print("\t"+"Mittelwerte über ungesehene Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model1"]["unseen_results"]["p"]["mean"],
                                                         glob_res["model1"]["unseen_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model1"]["unseen_results"]["r"]["mean"],
                                                         glob_res["model1"]["unseen_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model1"]["unseen_results"]["f1"]["mean"],
                                                         glob_res["model1"]["unseen_results"]["f1"]["std"],))
    print("\n\n")
    #plot_metrics_by_freq(glob_res["model1"]["freq_results"])
    #plot_prc(glob_res["model1"]["eval_by_delta"], no_skill=0.72)


    print("Modell 2 - Zuweisungen")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_test_results"]["p"]["mean"],
                                                         glob_res["model2"]["attr_test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_test_results"]["r"]["mean"],
                                                         glob_res["model2"]["attr_test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_test_results"]["f1"]["mean"],
                                                         glob_res["model2"]["attr_test_results"]["f1"]["std"],))
    print("\t"+"Mittelwerte über ungesehene Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_unseen_results"]["p"]["mean"],
                                                         glob_res["model2"]["attr_unseen_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_unseen_results"]["r"]["mean"],
                                                         glob_res["model2"]["attr_unseen_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model2"]["attr_unseen_results"]["f1"]["mean"],
                                                         glob_res["model2"]["attr_unseen_results"]["f1"]["std"],))
    print("\n\n")
    #plot_metrics_by_freq(glob_res["model2"]["attr_freq_results"])
    #plot_prc(glob_res["model2"]["attr_eval_by_delta"], no_skill=21.0)

    print("Modell 2 - NIL-Erkennung")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model2"]["nil_flag_test_results"]["p"]["mean"],
                                                         glob_res["model2"]["nil_flag_test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model2"]["nil_flag_test_results"]["r"]["mean"],
                                                         glob_res["model2"]["nil_flag_test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model2"]["nil_flag_test_results"]["f1"]["mean"],
                                                         glob_res["model2"]["nil_flag_test_results"]["f1"]["std"],))
    print("\n\n")
    #plot_prc(glob_res["model2"]["nil_flag_eval_by_delta"], no_skill=60.0)



    print("Modell 3")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model3"]["test_results"]["p"]["mean"],
                                                         glob_res["model3"]["test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model3"]["test_results"]["r"]["mean"],
                                                         glob_res["model3"]["test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model3"]["test_results"]["f1"]["mean"],
                                                         glob_res["model3"]["test_results"]["f1"]["std"],))
    print("\t"+"Mittelwerte über ungesehene Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["model3"]["unseen_results"]["p"]["mean"],
                                                         glob_res["model3"]["unseen_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["model3"]["unseen_results"]["r"]["mean"],
                                                         glob_res["model3"]["unseen_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["model3"]["unseen_results"]["f1"]["mean"],
                                                         glob_res["model3"]["unseen_results"]["f1"]["std"],))
    print("\n\n")
    #plot_metrics_by_freq(glob_res["model3"]["freq_results"])
    #plot_prc(glob_res["model3"]["eval_by_delta"], no_skill=72.0)


    print("-"*10+" Ergebnisse der Modell als Pipeline "+"-"*10)
    print()

    print("Modell 2 - Zuweisungen")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_test_results"]["p"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_test_results"]["r"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_test_results"]["f1"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_test_results"]["f1"]["std"],))
    print("\t"+"Mittelwerte über ungesehene Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_unseen_results"]["p"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_unseen_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_unseen_results"]["r"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_unseen_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["attr_unseen_results"]["f1"]["mean"],
                                                         glob_res["pipeline"]["model2"]["attr_unseen_results"]["f1"]["std"],))
    print("\n\n")
    #plot_metrics_by_freq(glob_res["pipeline"]["model2"]["attr_freq_results"])
    #plot_prc(glob_res["pipeline"]["model2"]["attr_eval_by_delta"], no_skill=21.0)

    print("Modell 2 - NIL-Erkennung")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["nil_flag_test_results"]["p"]["mean"],
                                                         glob_res["pipeline"]["model2"]["nil_flag_test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["nil_flag_test_results"]["r"]["mean"],
                                                         glob_res["pipeline"]["model2"]["nil_flag_test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model2"]["nil_flag_test_results"]["f1"]["mean"],
                                                         glob_res["pipeline"]["model2"]["nil_flag_test_results"]["f1"]["std"],))
    print("\n\n")
    #plot_prc(glob_res["pipeline"]["model2"]["nil_flag_eval_by_delta"], no_skill=50.0)



    print("Modell 3")
    print("\t"+"Mittelwerte über alle Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["test_results"]["p"]["mean"],
                                                         glob_res["pipeline"]["model3"]["test_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["test_results"]["r"]["mean"],
                                                         glob_res["pipeline"]["model3"]["test_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["test_results"]["f1"]["mean"],
                                                         glob_res["pipeline"]["model3"]["test_results"]["f1"]["std"],))
    print("\t"+"Mittelwerte über ungesehene Akteure:")
    print("\t"+"\t"+"Precision: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["unseen_results"]["p"]["mean"],
                                                         glob_res["pipeline"]["model3"]["unseen_results"]["p"]["std"],))
    print("\t"+"\t"+"   Recall: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["unseen_results"]["r"]["mean"],
                                                         glob_res["pipeline"]["model3"]["unseen_results"]["r"]["std"],))
    print("\t"+"\t"+"       F1: {:.2f}  ({:.2f})".format(glob_res["pipeline"]["model3"]["unseen_results"]["f1"]["mean"],
                                                         glob_res["pipeline"]["model3"]["unseen_results"]["f1"]["std"],))
    print("\n\n")
    #plot_metrics_by_freq(glob_res["pipeline"]["model3"]["freq_results"])
    #plot_prc(glob_res["pipeline"]["model3"]["eval_by_delta"], no_skill=72.0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please specify path to result-file...")
        exit()
    main(sys.argv[1])
    exit()
