#!/usr/bin/python3

import numpy as np
import random
import sys

from code import EmbeddingBootstrapper


def cos_dist(embeddings1, embeddings2):
    out = np.zeros((embeddings1.shape[0],embeddings2.shape[0]))
    for i in range(embeddings1.shape[0]):
        for j in range(embeddings2.shape[0]):
            enum = np.dot(embeddings1[i], embeddings2[j])
            denom = np.linalg.norm(embeddings1[i])*np.linalg.norm(embeddings2[j])
            out[i,j] =  1-(enum/denom)
    return out


def knn(query_entities, sim_matrix, map1, map2i, k=10):
    name_knns = []
    for id_ in query_entities:
        try:
            idx1 = map1[id_]
            nns = np.argsort(sim_matrix[idx1])
            kbest=[]
            for idx2 in nns[:k]:
                name2 = map2i[idx2]
                kbest.append(name2)
            name_knns.append((id_, kbest))
        except KeyError:
            pass
    return name_knns


def knn_self(entity_embeddings, emap, queries):
    # 
    emapi = dict([(i,l) for l,i in emap.items()])
    #
    sim_matrix = cos_dist(entity_embeddings, entity_embeddings)
    #
    knns = knn(queries, sim_matrix, emap, emapi, k=10)
    max_len = max([len(n) for _,nns in knns for n in nns])+2
    for name, nns in knns:
        print(" >    {:>15s}:\t{}".format(name, "\t".join([("{: <"+str(max_len)+"s}").format(s) for s in nns])))


def knn_word(entity_embeddings, emap, word_embeddings, wmap, queries):
    #
    emapi = dict([(i,l) for l,i in emap.items()])
    wmapi = dict([(i,l) for l,i in wmap.items()])
    #
    sim_matrix = cos_dist(entity_embeddings, word_embeddings)
    #
    knns = knn(queries, sim_matrix, emap, wmapi, k=10)
    max_len = max([len(n) for _,nns in knns for n in nns])+2
    for name, nns in knns:
        print(" >    {:<15s}:\t{}".format(name, "  ".join([("{:^"+str(max_len)+"s}").format(s) for s in nns])))


if __name__ == "__main__":
    if len(sys.argv) not in [2,3]:
        print("Please specify 'path1' or 'path1 path2'. Where path1 is entity_embedding file and path2 is the word_embedding file !!")
        exit()
    #
    entity_embeddings, emap = EmbeddingBootstrapper.load_fasttext(sys.argv[1], renorm=1.0)
    selected_queries = random.sample(emap.keys(), k=10)
    #
    if len(sys.argv) == 2:
        knn_self(entity_embeddings, emap, selected_queries)
        print()
    elif len(sys.argv) == 3:
        word_embeddings, wmap = EmbeddingBootstrapper.load_fasttext(sys.argv[2], renorm=1.0)
        knn_word(entity_embeddings, emap, word_embeddings, wmap, selected_queries)
        print()
    exit()
