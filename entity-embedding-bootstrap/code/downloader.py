"""
This file provides the Downloader class. This class is used to interact with the wikidata/wikimedia api and to download wikipedia pages from the german wikipedia.
"""

from bs4 import BeautifulSoup
import requests
import time
import math
import sys
import os


class Downloader(object):
    """
    This class is used for interacting with the wikimedia/wikidata api and downloading wikipeda pages.
    """
    def __init__(self):
        """
        Creates a new instance of the Downloader class. Holds a Requests.Session object.
        """
        self.session = requests.Session()
        self.sleep_after_request = 1.0
    
    def _query_entity_pages_(self, wikidata_ids):
        """
        Helper function. 
        Querys Wikidata api to find page-ids of wikipedia description page of specified wikimedia-entity-ids.
        """
        # Calcuate number of parst
        num_ids = len(wikidata_ids)
        num_parts = math.ceil(num_ids/49)  # maximum number of allowed entites per request is 50
        # Feed ids part-wise into the api
        wiki_pages = {}
        skipped = []
        for i in range(num_parts):
            s = int(i*49)
            e = min(num_ids, int(s+49))
            part_of_ids = wikidata_ids[s:e]
            # build query-params
            params = {
                "action": "wbgetentities",
                "format": "json",
                "props":"sitelinks",
                "ids": "|".join(part_of_ids),
                "sitefilter":"dewiki"}
            # request infos from wikidata
            r = self.session.get(url="https://www.wikidata.org/w/api.php",
                                params=params)
            time.sleep(self.sleep_after_request)
            if r.status_code!=200:
                return None
            # read response
            data = r.json()
            if "warning" in data:
                return None
            if "error"  in data:
                return None
            #
            for ent_id in data["entities"]:
                ent_data = data["entities"][ent_id]
                if ("missing" in ent_data):
                    skipped.append(ent_id)
                    continue
                if (len(ent_data["sitelinks"]) == 0):
                    skipped.append(ent_id)
                    continue
                page_title = ent_data["sitelinks"]["dewiki"]["title"]
                wiki_pages[page_title] = {"entity_id":ent_id,
                                          "page_title":page_title}
        assert(len(wiki_pages)+len(skipped) == len(wikidata_ids))
        return wiki_pages, skipped
    
    def _query_backlinks_(self, page_title):
        """
        Helper function. 
        Querys Wikimedia api to find page-ids wikipedia pages linking to the page with specified page_title.
        """
        # Prepare query-params
        params = {
            "action": "query",
            "format": "json",
            "list": "backlinks",
            "bllimit":"max",
            "blnamespace":"0",
            "bltitle": page_title}
        backlinks = {}
        # While response indicates more available links
        while True:
            # request in mediawiki
            r = self.session.get(url="https://de.wikipedia.org/w/api.php",
                                 params=params)
            time.sleep(self.sleep_after_request)
            if r.status_code!=200:
                return None
            # read request
            data = r.json()
            if "warning" in data:
                return None
            if "error"  in data:
                return None
            backlinks.update(dict([(bl["title"], {"page_id":str(bl["pageid"]),"page_title":bl["title"]}) for bl in data["query"]["backlinks"]]))
            # check if more data is available
            if "continue" not in data:
                break
            # add continue-param to query-parameters
            blcontinue = data["continue"]["blcontinue"]
            params["blcontinue"] = blcontinue
        return backlinks
    
    def _query_pageinfos_(self, pages):
        """
        Helper function.
        Query page infos (e.g. page_url and page_id) for the specified pages. Infos are stored into the specified pages.
        """
        # Calculate number of parts
        all_keys = list(pages.keys())
        num_keys = len(all_keys)
        num_parts = math.ceil(num_keys/49)  # maximum number of allowed entites is 50
        # Feed pages part wise into the api
        for i in range(num_parts):
            s = int(i*49)
            e = min(num_keys, int(s+49))
            part_of_pages = all_keys[s:e]
            # Prepare query-params
            params = {
                "action": "query",
                "format": "json",
                "prop": "info",
                "inprop":"url",
                "titles":"|".join([pages[k]["page_title"] for k in part_of_pages]),
            }
            # request in mediawiki
            r = self.session.get(url="https://de.wikipedia.org/w/api.php",
                                 params=params)
            time.sleep(self.sleep_after_request)
            if r.status_code!=200:
                return None
            # read response
            data = r.json()
            if "warning" in data:
                return None
            if "error"  in data:
                return None
            for page_id, page_info in  data["query"]["pages"].items():
                pages[page_info["title"]].update({"page_id":page_id,
                                                  "page_url":page_info["canonicalurl"]})
        return pages
    
    def download_page(self, page_url):
        """
        Downloads the wikipedia page with the specified page_url.
        Returns a BeautifulSoup html parse of the page.
        """
        # download
        r = self.session.get(url=page_url)
        time.sleep(self.sleep_after_request)
        if r.status_code!=200:
            return None
        # parse
        soup = BeautifulSoup(r.content, 'html.parser')
        return soup
    
    def create_new_jobs(self, wikidata_ids):
        """
        For each wikidata-entity-id in the specified wikidata_ids a dict is created.
        This dict holds information about the wikipedia description page and pages linking there.
        If no description page exists for an entitiy, the entity is skipped.
        Returns a list of dictionarys and a list of skipped entities.
        """
        print(" > Creating new jobs ...")
        # Find Wikipedia-Page for every Wikimedia-Entity
        pages, skipped = self._query_entity_pages_(wikidata_ids)
        # Find Infos about each page
        if self._query_pageinfos_(pages) is None:
            print("DEBUG_ERROR: first query_pageinfos")
            return None
        # Find and add Wikipedia-Backlinks
        num_pages = len(pages)
        i=0
        for target_page in pages.values():
            # Find list of pages linking to target-page
            source_pages = self._query_backlinks_(target_page["page_title"])
            # Find infos about the linking pages
            if self._query_pageinfos_(source_pages) is None:
                print("\nDEBUG-WARNING: first query_pageinfos")
                continue
            # Add link-infos to target page info
            target_page["back_links"] = list(source_pages.values())
            # Print percentage of finished work
            sys.stdout.write(u"\u001b[1000D")
            sys.stdout.flush()
            sys.stdout.write(" >     {:6.2f} %".format(100*(i+1)/num_pages))
            sys.stdout.flush()
            i += 1
        print()
        return list(pages.values()), skipped
