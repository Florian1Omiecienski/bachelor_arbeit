"""
In this file the DataManger clas is provided. This Class is used to organize the I/O operations for the bootstrapping process.
"""

from bs4 import BeautifulSoup
import shutil
import json
import math
import os


class DataManager(object):
    """
    This class organizes the data i/o for the entitiy embedding creation.
    It provides I/O methods for a database of downloaded wikipedia files.
    It provides I/O methods for the temporary files created during the process (job and index file).
    It provides a method for writing the preprocessed data-files.
    """
    def __init__(self, working_dir, run_id):
        self.working_dir = working_dir
        self.run_id = run_id
        #
        self.job_file_path = os.path.join(self.working_dir, self.run_id+".jobs")
        self.index_file_path = os.path.join(self.working_dir, self.run_id+".index")
        #
        self.database_path = os.path.join(self.working_dir, "database")
        self.database_meta_path = os.path.join(self.database_path, "download_times.meta")
        self.database_root_path = os.path.join(self.database_path, "pages")
        self.suffix_length = 3
        #
        self.entity_file_dir = os.path.join(self.working_dir, "entity_files")
    
    def load_index(self):
        """
        Loads the index stored in the index file.
        """
        with open(self.index_file_path, "tr") as ifile:
            return int(ifile.readline().strip())
    
    def load_jobs(self):
        """
        Loads the jobs stored in the jobs file.
        """
        with open(self.job_file_path, "tr") as ifile:
            return json.load(ifile)
    
    def write_index(self, idx):
        """
        Writes the index to file.
        """
        with open(self.index_file_path, "tw") as ofile:
            ofile.write(str(int(idx)))
        return None
    
    def write_jobs(self, jobs):
        """
        Writes the jobs to file.
        """
        with open(self.job_file_path, "tw") as ofile:
            json.dump(jobs, ofile)
        return None
    
    def init_new_database(self):
        """
        Deletes old database and creates a new database directory as well as a meta-data file.
        """
        # (empty old and) create new database directory
        if os.path.isdir(self.database_path):
            shutil.rmtree(self.database_path)
        os.mkdir(self.database_path)
        # create meta_file
        with open(self.database_meta_path, "tw") as ofile:
            pass
        # create root_dir
        os.mkdir(self.database_root_path)
        return None
    
    def store_in_database(self, page_id, bs4_page, meta):
        """
        Stores the specified BeautifullSoup html page with the given meta information into the database.
        The filename is the page_id.
        """
        # find place to store in database
        tpath = self.database_root_path
        if len(page_id) > self.suffix_length:
            prefix = page_id[:-self.suffix_length]
            # find path to store
            for c in reversed(list(prefix)):
                tpath = os.path.join(tpath, c)
                if not os.path.isdir(tpath):
                    os.mkdir(tpath)
        # store in database
        opath = os.path.join(tpath, page_id+".html")
        with open(opath, "tw") as ofile:
            ofile.write(str(bs4_page))
        # append meta-info to meta_file
        with open(self.database_meta_path, "ta") as ofile:
            ofile.write("{}\t{}\t{}\t{}\n".format(page_id, meta["page_title"], meta["page_url"], meta["time_stamp"]))
        return
    
    def load_from_database(self, page_id):
        """
        Reads a html page with specified page id from database.
        Returns a BeautifulSoup object.
        """
        # find place to load in database
        tpath = self.database_root_path
        if len(page_id) > self.suffix_length:
            prefix = page_id[:-self.suffix_length]
            # find path to store
            for c in reversed(list(prefix)):
                tpath = os.path.join(tpath, c)
        ipath = os.path.join(tpath, page_id+".html")
        # load file
        with open(ipath, "tr") as ifile:
            soup = BeautifulSoup(ifile, 'html.parser')
        return soup
    
    def is_in_database(self, page_id):
        """
        Returns True if a page with specified page_id is in the database.
        Else False is returned.
        """
        # find place to check in database
        tpath = self.database_root_path
        if len(page_id) > self.suffix_length:
            prefix = page_id[:-self.suffix_length]
            # find path to store
            for c in reversed(list(prefix)):
                tpath = os.path.join(tpath, c)
        # store in database
        query_path = os.path.join(tpath, page_id+".html")
        return os.path.isfile(query_path)
    
    def write_entity_file(self, entity_data):
        """
        Writes the specified entity_data to file.
        """
        if not os.path.isdir(self.entity_file_dir):
            os.mkdir(self.entity_file_dir)
        opath = os.path.join(self.entity_file_dir, entity_data["entity_id"]+".json")
        with open(opath, "tw") as ofile:
            json.dump(entity_data, ofile)
        return
