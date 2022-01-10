#!/usr/bin/python3
"""
This file provides a python3 script. This script trains the entity embeddings from some prepared data.
"""

import argparse
import time
import os

from code import DataManager
from code import EmbeddingBootstrapper


def parse_args():
    """
    Handle the commandline arguments.
    """
    # setup argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('entity_directory', type=str,
                        help='')
    parser.add_argument('fasttext_path', type=str, 
                        help='Specifies the path to a text file. File holds the FastText-Vectors.')
    parser.add_argument('word_counts', type=str, 
                        help='Specifies the path to a text file. File holds unigram word counts.')
    parser.add_argument('output_file', type=str,
                        help='')
    parser.add_argument('--append', '-a', action="store_true",
                        help='')

    args = parser.parse_args()
    # check arguments
    # return arguments
    return vars(args)


def main():
    # train parameters
    margin  = 0.3
    info_lr = 0.1
    link_lr = 0.1
    info_epochs = 200
    link_epochs = 300
    npos_sampels = 5
    nneg_sampels = 10
    smoothing_exp = 0.70
    # preprocessing parameter
    renorm_fsttxt = 1.0
    include_mention=False
    # Regularization parameters
    one_norm=1.0
    sec_param=None
    max_norm=None
    # Parse commandline arguments
    args = parse_args()
    # Load FastText exmbeddings
    fsttx, fsttxt_map = EmbeddingBootstrapper.load_fasttext(args["fasttext_path"],
                                                            renorm=renorm_fsttxt)
    # Load unigram word counts from file
    word_counts = EmbeddingBootstrapper.load_words_counts(args["word_counts"], word_map=fsttxt_map)
    # Create model for bootstrapping the embeddings
    bootstraper = EmbeddingBootstrapper(fsttx, fsttxt_map, word_counts, entity_embeddings=None, entity_map=None,
                                        smoothing_exp=smoothing_exp, 
                                        margin=margin, 
                                        num_pos_sampels=npos_sampels,
                                        num_neg_sampels=nneg_sampels, 
                                        one_norm=one_norm,
                                        sec_param=sec_param,
                                        max_norm=max_norm)
    # Load data for the first training round
    info_data = EmbeddingBootstrapper.load_wiki_info_data(args["entity_directory"])
    print(" > Loaded description-page infos for {} entities.".format(len(info_data)))
    # Train on info-page-data
    bootstraper.train(info_data, num_epochs=info_epochs, learning_rate=info_lr)
    # Load data for the second training round
    link_data = EmbeddingBootstrapper.load_wiki_link_data(args["entity_directory"],
                                                          include_mention=include_mention)
    print(" > Loaded linking-page infos for {} entities.".format(len(link_data)))
    # Train on linking-page-data
    bootstraper.train(link_data,
                      num_epochs=link_epochs,
                      learning_rate=link_lr,
                      stop_iterations=20)
    # Store trained-embeddings
    bootstraper.store_embeddings(args["output_file"])


if __name__ == "__main__":
    main()
