#!/usr/bin/python3
"""
This file holds the code for a class used for handling data-i/o as well as other data related operations.
The DataHandler class has methods for loading the original mardy data, loading embeddings files, as well as methods for spliting the dataset.

This file was created for the 'Bachelor Arbeit' from Florian Omiecienski.
Autor: Florian Omiecienski
"""
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import random
import math
import pickle
import json
import os

from .data import Document


class DataHandler(object):
    """
    """
    @staticmethod
    def load_mardy_data(article_path, claim_path, entity_path, actor_mapping_path, og_anno_path, known_actors):
        """
        Loads the mardy data from the specified paths. Only documents are loaded, which contain
        normalized actor information as well as actor-mapping information.
        Returns a list of document objects.
        article_path: Path to a directory, containing the newspaper-documents (json files).
        claim_path: Path to a file, containing the claim-annotations with normalized actor-names (line separated json file).
        entity_path: Path to a file, containing the manually added entity annotations (line separated json file).
        actor_mapping_path: Path to a file, containing actor mapping data for each document (line separated json file).
        og_anno_path:  Path to a file, containing the original claim with un-normalized actor-names (line separated json file).
        known_actors: A set of known WikiData-IDS, an actor will be called NIL-Actor if the given WikiData-ID is not in this set.
        """
        # Load data
        articles            = DataHandler._load_articles(article_path)
        claim_annos         = DataHandler._load_jsonl(claim_path)
        entity_annos        = DataHandler._load_jsonl(entity_path)
        actor_mapping_annos = DataHandler._load_jsonl(actor_mapping_path)
        original_annos      = DataHandler._load_jsonl(og_anno_path)
        
        # Sort data by document_id
        actor_mapping_by_docid  = DataHandler._create_actor_mapping(actor_mapping_annos, known_actors)
        claim_annos_by_docid    = DataHandler._sort_by_doc_id(claim_annos)
        entity_annos_by_docid   = DataHandler._sort_by_doc_id(entity_annos)
        original_annos_by_docid = DataHandler._sort_by_doc_id(original_annos)
        print("Found {} documents in total".format(len(articles)))
        print("Found Actor-Mapping annotations for {} documents".format(len(actor_mapping_by_docid)))
        print("Found Claim annotations for {} documents".format(len(claim_annos_by_docid)))
        print("Found Entity annotations for {} documents".format(len(entity_annos_by_docid)))
        # Select mardy-document with available actor-mapping
        doc_ids = set(actor_mapping_by_docid.keys()).intersection(set(articles.keys()))
        print("Working with {} documents !!!".format(len(doc_ids)))
        # Create Document objects for all selected documents
        documents = []
        for doc_id in doc_ids:
            # Get annos by doc_id
            article = articles[doc_id]
            claim_annos = claim_annos_by_docid[doc_id] if doc_id in claim_annos_by_docid else []
            entity_annos = entity_annos_by_docid[doc_id] if doc_id in entity_annos_by_docid else []
            actor_mapping = actor_mapping_by_docid[doc_id] if doc_id in actor_mapping_by_docid else {}
            original_annos = original_annos_by_docid[doc_id] if doc_id in original_annos_by_docid else []
            # Create a document 
            new_doc = Document.from_mardy_data(doc_id, article, claim_annos, original_annos, entity_annos, actor_mapping)
            if new_doc is None:
                print("Warning: Document is skipped")
            documents.append(new_doc)
        # Return all documents
        return documents
    
    @staticmethod
    def load_embeddings(path, dim=300, create_unk=False, renorm=None):
        """
        Load embeddings from specified path. The embeddings are expected to have dimensionlity of dim.
        If renorm=int(), all embeddings are re-normalized to this value.
        If create_unk=True, adds a zero-vector associated with the UNK-Token.
        Returns a tuple: (Numpy.array, dict) containing the embeddings data and the token-index-mapping.
        """
        vecs = []
        map_ = dict()
        # Add special tokens to the map
        if create_unk is True:
            map_["<UNK>"] = 0
        # Load embeddings-data
        with open(path, "tr") as ifile:
            for line in ifile:
                values = line.rstrip().split(" ")
                if len(values) == dim+1:
                    name = values[0]
                    values = [float(f) for f in values[1:]]
                    map_[name] = len(map_)
                    vecs.append(np.array([values,]))
        vecs =  np.row_stack(vecs)
        # Renorm all embeddings to the value given
        if renorm is not None:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs *= (renorm/norms)
        # Add zero-vectors for the special tokens
        if create_unk is True:
            vecs = np.row_stack([np.zeros((1, dim)), vecs])
        # Sanity-Check and return
        assert(vecs.shape[0] == len(map_))
        assert(vecs.shape[1] == dim)
        return vecs, map_
    
    @staticmethod
    def split(documents, train_ratio, dev_ratio, test_ratio, randomize=True):
        """
        Creates a train/dev/test split for the given documents.
        If randomize is True, data is shuffeld befor splitting.
        Return a triple of lists. Each list is a split.
        """
        assert(math.isclose((train_ratio+dev_ratio+test_ratio), 1))
        n = len(documents)
        i1 = int(n*train_ratio)
        i2 = int(n*(train_ratio+dev_ratio))
        if randomize is True:
            documents = list(documents)
            random.shuffle(documents)
        return documents[:i1], documents[i1:i2], documents[i2:]
    
    @staticmethod
    def stratified_kfold_iterator(documents, num_folds, dev_ratio=0.25):
        """
        Splits a set of documents into num_folds many data-splits.
        Uses the stratified kfold method form sk-multilearn.
        """
        documents = list(documents)
        random.shuffle(documents)
        # Turn actor-doc-sets into a multiclass label-set
        mlb = MultiLabelBinarizer()
        data = [(d, [a.wikidata_id for a in d.actors if a.is_nil is False]) for d in documents]
        # Replace empty label-sets with a set containing one label named "EMPTY"
        for i in range(len(data)):
            inst = data[i]
            if len(inst[1])==0:
                y = ["EMPTY", ]
                data[i] = (inst[0], y)
        # Turn labels into multi-hot-vecs using scipy
        X,y = zip(*data)
        y_labels = mlb.fit_transform(y)
        # Do stratified kfold
        kfold = IterativeStratification(n_splits=num_folds)
        for remain, test in kfold.split(X,y_labels):
            Xtest = [X[i] for i in test]
            # Split remaining data into train and dev
            kfold2 = IterativeStratification(n_splits=2, sample_distribution_per_fold=[dev_ratio, 1.0-dev_ratio])
            Xr, yr = [X[i] for i in remain], y_labels[remain]
            train, dev = next(kfold2.split(Xr, yr))
            Xtrain = [Xr[i] for i in train]
            Xdev = [Xr[i] for i in dev]
            yield Xtrain, Xdev, Xtest
    
    
    @staticmethod
    def write_documents(documents, path):
        """
        Stores the set of specified documents to the path.
        """
        with open(path, "wb") as ofile:
            pickle.dump(documents, ofile)
        return
    
    @staticmethod
    def load_documents(path):
        """
        Loads a set of documents from the specified path.
        """
        with open(path, "rb") as ifile:
            return pickle.load(ifile)
    
    @staticmethod
    def _load_jsonl(path):
        """
        Loads jsonl format files
        """
        elements = []
        with open(path) as file:
            for line in file:
                elements.append(json.loads(line))
        if len(elements) == 1:
            return elements[0]
        return elements
    
    @staticmethod
    def _load_articles(path_to_folder, fextension="json"):
        """
        Loafs all debatenet articles from specified directory
        """
        articles = {}
        article_files = os.listdir(path_to_folder)
        article_files = [e for e in article_files if e.split(".")[-1]==fextension]
        list_of_paths = [str(os.path.join(path_to_folder, e)) for e in article_files]
        for path in list_of_paths:
            art = DataHandler._load_jsonl(path)
            name = int(path.split("/")[-1].split(".")[0])
            articles[name] = art
        return articles
    
    @staticmethod
    def _sort_by_doc_id(list_):
        """
        Sorts the list elements into a dict. Doc-IDs are used as keys.
        """
        out = dict()
        for e in list_:
            try:
                out[int(e["doc_id"])].append(e)
            except KeyError:
                out[int(e["doc_id"])] = [e, ]
        return out
    
    @staticmethod
    def _create_actor_mapping(actor_mapping_annotations, known_actors):
        """
        Sorts actor-mapping-data by doc-id and ignores unknown_actors.
        """
        mappings_by_docid = {}
        for anno in actor_mapping_annotations:
            if "wd" in anno:
                wd_id = anno["wd"]
                if wd_id in known_actors:   # filter out actor-entity-ids which are not in the given KB
                    name = anno["ne"].strip(" ")
                    doc_ids = [int(di) for di in anno["docs"]]
                    for di in doc_ids:
                        if di not in mappings_by_docid:
                            mappings_by_docid[di] = {}
                        mappings_by_docid[di][name] = wd_id
        return mappings_by_docid

