#!/usr/bin/python3
"""
This file provides a python3 script. This script downloads all information and wikipedia pages needed for the training of the entity embeddings.
"""

import argparse
import time
import sys
import os

from code import Downloader
from code import DataManager


def _check_arguments_(args):
    """
    Checking commandline arguments.
    """
    if not os.path.isdir(args.working_dir):
        print(" > INVALID-ARGUMENTS: Specified WORKING_DIR is not a directory.")
        return False
    # check arguments
    e = (args.entities is not None)
    r = (args.reinit is True)
    # check if a job files exists
    j1 = os.path.isfile(os.path.join(args.working_dir, args.run_id+".jobs"))
    j2 = os.path.isfile(os.path.join(args.working_dir, args.run_id+".index"))
    j = j1 and j2
    if (not j) and (j1 or j2):
        print(" > ERROR: Detected inconsistent job-files. Delete the files RUN_ID.jobs and RUN_ID.index in WORKING_DIR to fix.")
        return False
    #
    d1 = os.path.isdir(os.path.join(args.working_dir, "database"))
    d2 = os.path.isfile(os.path.join(os.path.join(args.working_dir, "database"),"download_times.meta"))
    d3 = os.path.isdir(os.path.join(os.path.join(args.working_dir, "database"),"pages"))
    d = (d1 and d2 and d3)
    if (not d) and (d1 or d2 or d3):
        print(" > ERROR: Detected inconsistent databse. To create a new database use only -r flag.")
        return False
    #
    if j and r:
        print(" > INVALID-ARGUMENTS: Found open jobs for run_id={}. Delete the jobs file first to (re)initialize a new database.")
        return False
    if (not j) and e and r:
        return True
    if (j or e) and (not d):
        print(" > INVALID-ARGUMENTS: No database found. Befor the first crawl, call with -r flag only.")
        return False
    if j and e:
        print(" > INVALID-ARGUMENTS: Found open jobs for run_id={}. Delete the jobs file first to restart with this run_id. Or remove the -e flag to continue on open jobs.".format(args.run_id))
        return False
    if (not e) and (not j):
        print(" > INVALID-ARGUMENTS: No open jobs found for run_id={}. Specify -e to start new jobs.".format(args.run_id))
        return False
    if e:
        if os.path.isfile(args.entities) is False:
            print(" > INVALID-ARGUMENT: Specified path to entity list is invalid !!")
            return False
    return True


def parse_args():
    """
    Handle the commandline arguments.
    """
    # setup argparse
    parser = argparse.ArgumentParser(description='Downloads data from Wikipedia. Given a list of Wikimedia-Entities there description pages and all other pages that link to these description pages are downloaded and stores in a database-directory. After the first usage the database can be reused by NOT specifing the -R flag. The run_id specifies one downloading-task. After this task is finished you can start another task with different run_id or use the same run_id again with the -r(estart) flag. But in this case all data from the first task (excluding the database) will be overwritten.')
    parser.add_argument('working_dir', type=str,
                        help='Specifies the path to a directory. This directory will hold all output. Make sure no other directory with this name exists otherwise it will be deleteted !!')
    parser.add_argument('run_id', type=str,
                        help='Specifies a uniq identifier for the crawl.')
    parser.add_argument('--entities', '-e', type=str, default=None,
                        help='Speciefies the path to a text file. File holds Wikidata entities line-separated.')
    parser.add_argument('--reinit', '-r', action="store_true",
                        help='Flag that specifies if a new database should be created. Use this flag ONLY for the first crawl.')
    args = parser.parse_args()
    # check arguments
    if _check_arguments_(args) is False:
        exit()
    # load entities from specified file
    entities = None
    if args.entities is not None:
        with open(args.entities, "tr") as ifile:
            entities = []
            i = 1
            for line in ifile:
                if line in ["", "\n"]:
                    continue
                if line[0]!="Q":
                    print("WARNING: line {} does not contain a valid WikiData-Entity-ID [{}] in the specified file!".format(i, line.rstrip("\n")))
                    continue
                name = line.rstrip("\n")
                entities.append(name)
                i += 1
    # return arguments
    return args.working_dir, args.run_id, entities, args.reinit


def main():
    """
    Run the script.
    Step1: Download meta-data or load job-file
    Step2: Download wikipedia pages to database
    """
    # Parse arguments
    working_dir, run_id, wm_entities, new_database_flag = parse_args()
    # Create Classes
    downloader = Downloader()
    data_manager = DataManager(working_dir=working_dir,
                               run_id=run_id)
    ### Step 1 (preparing the jobs of the crawl)
    if wm_entities is not None:
        # Download and save meta-data (page_infos, link_infos)
        jobs, skipped = downloader.create_new_jobs(wm_entities)
        print(" > No description pages found for:", skipped)
        jobs.sort(key=lambda x:len(x["back_links"]))   # sort by expected workload (increasing)
        # Write the inital job-files
        index = 0
        data_manager.write_jobs(jobs)
        data_manager.write_index(index)
    else:
        # Load the job files
        print(" > Loading jobs ...")
        jobs = data_manager.load_jobs()
        index = data_manager.load_index()
    print(" >     Total-Jobs:", len(jobs))
    print(" >     Job-Index:", index, end="\n\n\n")
    # Init a new database
    if new_database_flag is True:
        print(" > Initialising new database ...")
        data_manager.init_new_database()
    
    ### Step 2 (downloading wiki pages)
    num_jobs = len(jobs)
    while index < num_jobs:
        job = jobs[index]
        num_skipped = 0
        print(" > Downloading data for [{}]".format(job["entity_id"]))
        # Download description-page
        page_id = job["page_id"]
        page_url = job["page_url"]
        page_title = job["page_title"]
        if not data_manager.is_in_database(page_id):
            print(" >     Downloading Info-Page ...")
            time_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
            bs4page = downloader.download_page(page_url)
            meta = {"page_id":page_id,
                    "page_url":page_url,
                    "page_title":page_title,
                    "time_stamp":time_stamp}
            data_manager.store_in_database(page_id, bs4page, meta)
        else:
            num_skipped += 1
        # Download linking-pages
        print(" >     Downloading Link-Pages ...")
        num_links = len(job["back_links"])
        for j in range(num_links):
            src_page = job["back_links"][j]
            page_id = src_page["page_id"]
            page_url = src_page["page_url"]
            page_title = src_page["page_title"]
            if not data_manager.is_in_database(page_id):
                time_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
                bs4page = downloader.download_page(page_url)
                meta = {"page_id":page_id,
                        "page_url":page_url,
                        "page_title":page_title,
                        "time_stamp":time_stamp}
                data_manager.store_in_database(page_id, bs4page, meta)
            else:
                num_skipped += 1
            # print precentage of already downloaded
            sys.stdout.write(u"\u001b[1000D")
            sys.stdout.flush()
            sys.stdout.write(" >     {:6.2f} %".format(100*(j+1)/num_links))
            sys.stdout.flush()
        print()
        # Set Job-Index
        index += 1
        data_manager.write_index(index)
        print(" > Index is now at {:>d} / {:<d}".format(index, len(jobs)))
        print(" > Number of skipped-Sites {:>d} / {:<d}".format(num_skipped,num_links+1))
        print(" > Its now [{}]".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print("\n\n")


if __name__ == "__main__":
    main()
