#!/usr/bin/python3

from bs4 import BeautifulSoup
import numpy as np
import argparse
import glob
import os


from code import DataExtractor


def list_files(path):
    return [f for f in glob.glob(path + "**/*.html", recursive=True)]


def load_html(path):
    with open(path, "tr") as ifile:
        soup = BeautifulSoup(ifile, 'html.parser')
    return soup


def load_fasttext_vocab(path, size=300):
    wmap = dict()
    with open(path, 'tr', encoding='utf-8') as ifile:
        #
        for line in ifile:
            #
            values = line.rstrip().split(' ')
            #
            if len(values) < size+1:
                continue
            #
            word = values[0]
            if word not in wmap:
                wmap[word] = 0
            #
    return wmap


def write_counts(path, counter):
    counts = [(t,c) for t,c in counter.items()]
    counts.sort(key=lambda x:x[1])
    with open(path, "tw") as ofile:
        for t,c in counts:
            ofile.write("{} {}\n".format(t, c))
    return None


def str2int(string):
    lookup = {"K":1e3,
              "M":1e6}
    if string.isnumeric():
        out = int(string)
    else:
        num, mag = string[:-1], string[-1]
        num = int(num)
        mag = lookup[mag]
        out = int(num*mag)
    return out


def parse_args():
    """
    Handle the commandline arguments.
    """
    # setup argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('working_dir', type=str,
                        help='')
    parser.add_argument('fasttext_path', type=str, 
                        help='Specifies the path to a text file. File holds the FastText-Vectors.')
    parser.add_argument('output_file', type=str,
                        help='')
    parser.add_argument('--num_articles', "-n", type=str, default="100K",
                        help='')
    args = parser.parse_args()
    # check arguments
    num_articles = str2int(args.num_articles)
    # return arguments
    return args.working_dir, args.fasttext_path, args.output_file, num_articles


if __name__ == "__main__":
    wd_path, ft_path, opath, num_articles = parse_args()
    #
    db_path = os.path.join(wd_path, "database/pages/")
    max_num_articles = num_articles
    #
    fsttxt_map = load_fasttext_vocab(ft_path)
    #
    data_extractor = DataExtractor("de_core_news_sm", 0)
    all_articel_paths = list_files(db_path)
    #
    N = min(max_num_articles, len(all_articel_paths))
    for i in range(N):
        print(i,"/",N)
        article_path = all_articel_paths[i]
        page = load_html(article_path)
        text = data_extractor.html2text(page)
        if text is None:
            print("Page [{}] is None !!".format(article_path))
            continue
        tokens = data_extractor.preprocess_text(text)
        for t in tokens:
            try:
                fsttxt_map[t] += 1
            except KeyError:
                pass
    #
    write_counts(opath, fsttxt_map)
