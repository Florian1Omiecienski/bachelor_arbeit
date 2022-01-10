#!/usr/bin/python3
"""
This file provides a python3 script. This script extracts and prepares the downloaded wikipedia data for the training of the entity embeddings.
"""

import argparse
import time
import os

from code import DataManager
from code import DataExtractor


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(description='Prepares the data needed for the bootstrap_embeddings.py script. Run this script on the output of the crawl_wikipedia.py script.')
    parser.add_argument('working_dir', type=str,
                        help='Specifies the path to a workign directory created with the crawl_wiki.py script.')
    parser.add_argument('run_id', type=str,
                        help='Specifies the uniq identifier of the craw that should be processed. (A .jobs file must exist for that crawl). ')
    parser.add_argument('--window_size', '-e', type=int, default=10,
                        help='Specifies window-size (applied on both sides). Must be an integer. Default is 10.')
    parser.add_argument('--start', '-s', type=int, default=None,
                        help='Specifies at which index to start. Usually the index the program was stoped at an earlier run. Must be an integer. Default is None.')
    parser.add_argument('--skip', '-S', action="store_true",
                        help='If specified, entities which have an output file already are skipped.')
    parser.add_argument('--noinfo', '-v', type=int, default=None,
                        help='If specified less infos are printed.')
    args = parser.parse_args()
    # check arguments
    if not os.path.isdir(args.working_dir):
        print(" > INVALID-ARGUMENT: Invalid working directory specified.")
        exit()
    return args.working_dir, args.run_id, args.window_size, args.start, args.noinfo, args.skip


def main():
    """
    Runs the script.
    """
    #
    wait_for_data_time = 120
    spacy_path = "de_core_news_sm"
    # parse arguments
    working_dir, run_id, window_size, start, noinfo_flag, skip_flag = parse_args()
    # create classes
    data_manager = DataManager(working_dir=working_dir,
                               run_id=run_id)
    data_extractor = DataExtractor(spacy_path=spacy_path,
                                  window_size=window_size)
    # checks if a job file exists
    if not os.path.isfile(data_manager.job_file_path):
        print(" > INVALID-ARGUMENT: No job-files found for the specified job-id")
        exit()
    # set the index to start with
    process_index = 0 if start is None else start
    # runs until broken with KeyboarInterrupt
    try:
        last_waiting_flag = False
        jobs = data_manager.load_jobs()
        while True:
            # load the current index of the downloader script
            index = data_manager.load_index()
            if last_waiting_flag is False:
                print(" > Jobs todo:", len(jobs))
                print(" > Downloader index:", index)
                print(" > Processing index:", process_index, end="\n\n")
            # if all available data is process, then wait
            if index == process_index:
                print(" > Waiting ...", end="\n")
                last_waiting_flag = True
                # wait
                time.sleep(wait_for_data_time)
                # check again
                continue
            print("\n")
            # else process all available data
            for i in range(process_index, index):
                # for the current entitiy
                job = jobs[i]
                entity_id = job["entity_id"]
                print(" > Processing job {}  [{}, {}] ...".format(i, entity_id, job["page_title"]))
                # check if entity is already processed
                if skip_flag is True:
                    entity_path = os.path.join(data_manager.entity_file_dir,entity_id+".json")
                    if os.path.isfile(entity_path):
                        print(" >    Skipped because the file [{}] already exisits.".format(entity_path))
                        continue
                # process description page
                pageid = job["page_id"]
                target_url = job["page_url"]
                info_page = data_manager.load_from_database(pageid)
                info_text = data_extractor.html2text(info_page)
                if info_text is None:
                    if noinfo_flag is False:
                            print(" >    Error on InfoPage of [{}]".format(job["page_title"]))
                    continue
                info_tokens = data_extractor.preprocess_text(info_text)
                # process linking pages
                linkdata = []
                for linkpage in job["back_links"]:
                    pageid = linkpage["page_id"]
                    src_page = data_manager.load_from_database(pageid)
                    mention_links = data_extractor.find_links(src_page, target_url)
                    # debug-start
                    if(len(mention_links)==0):
                        if noinfo_flag is False:
                            print(" >    No links found at all for [{}] ".format(linkpage["page_title"]))
                        continue
                    # debug-end
                    src_text = data_extractor.html2text_marked(src_page, mention_links)
                    if src_text is None:
                        if noinfo_flag is False:
                            print(" >    Error on LinkPage of  [{}]".format(linkpage["page_title"]))
                        continue
                    src_tokens = data_extractor.preprocess_text(src_text)
                    mention_spans = data_extractor.find_mentions(src_tokens)
                    mention_data = data_extractor.extract_context_windows(src_tokens, mention_spans)
                    linkdata.extend(mention_data)
                # merge data of entity
                entitiy_data = {"entity_id":entity_id,
                                "description_tokens":info_tokens,
                                "link_data":linkdata}
                # write data of entity
                data_manager.write_entity_file(entitiy_data)
                print(" > Finished index", i,"[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S')))
            # set index of processing
            process_index = index
            print("\n")
    except KeyboardInterrupt:
        print(" > User Interrupt !!", end="\n\n")


if __name__ == "__main__":
    main()
