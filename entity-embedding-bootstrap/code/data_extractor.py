"""
This file provides the DataExtractor class. This Class provides methods for the preprocessing of the wikipedia articels.
"""

from bs4.element import NavigableString, Comment, Tag
from spacy.symbols import ORTH
from urllib.parse import urlparse
import spacy


class DataExtractor(object):
    """
    This class is used to preprocess the downloaded wikipedia pages and to extract the data needed to bootstrap entity embeddings.
    Spacy is used to tokenize and to normalize the wikipedia pages.
    """
    def __init__(self, spacy_path, window_size):
        """
        Creates a DataExtractor with spacy_path as the spacy model. 
        The extract_context_windows method extract window_size words on both sides of a mention
        """
        self.nlp = spacy.load(spacy_path)
        self.window_size = window_size
        # Make spacy ignore the mention tag at tokenization
        special_case = [{ORTH: "<MENTION>"}]
        self.nlp.tokenizer.add_special_case("<MENTION>", special_case)
        special_case = [{ORTH: "<\MENTION>"}]
        self.nlp.tokenizer.add_special_case("<\MENTION>", special_case)
    
    @staticmethod
    def html2text(bs4page):
        """
        Takes a BeautifulSoup object of a wikipedia page and returns plain text.
        The text is extracted only from paragraphs (<p>) which occure directly on the content plain of the page.
        This means that informations like image descriptions, tabels, etc. are not taken into account.
        """
        # find content div in html
        content_div = bs4page.find(name="div", class_="mw-parser-output")
        if content_div is None:
            return None
        # select all paragraphs
        info_paragraphs = content_div.find_all(name="p", recursive=False)
        # remove citings from text like "[1]", "[2]"  # maybe not necessary
        for p in info_paragraphs:
            for sup in p.find_all("sup", {'class':'reference'}):
                sup.decompose()
        # merge paragraph
        text = ""
        for p in info_paragraphs:
            text +=p.get_text()
        #
        return text
    
    @staticmethod
    def find_links(bs4page, target_url):
        """
        Given a BeautifullSoup html parse and a traget url string, finds all links in the parse that link to the target url.
        """
        # find target page-specifier
        target = urlparse(target_url).path.split("/wiki/")
        target = target[-1]
        # find all links that point to the target
        links = []
        for link in bs4page.find_all("a", href=True, recursive=True):
            inst = urlparse(link["href"]).path.split("/wiki/")
            inst = inst[-1]
            if inst==target:
                links.append(link)
        return links
    
    @staticmethod
    def html2text_marked(bs4page, links_to_mark):
        """
        Takes a BeautifulSoup object of a wikipedia page and a list of bs4 Tags (links). 
        Returns plain text where the mentions in links_to_mark are enclode in the markers <MENTION></MENTION>.
        The text is extracted only from paragraphs (<p>) which occure directly on the content plain of the page.
        This means that informations like image descriptions, tabels, etc. are not taken into account.
        """
        content_div = bs4page.find(name="div", class_="mw-parser-output")
        if content_div is None:
            return None
        text = ""
        for par in content_div.find_all(name="p", recursive=False):
            # remove citings from html (e.g. [1], [2])
            for sup in par.find_all("sup", {'class':'reference'}):
                sup.decompose()
            # extract all paragraph text
            for child in par:
                if isinstance(child, NavigableString):
                    text += str(child)
                elif isinstance(child, Tag):
                    if child not in links_to_mark:
                        text += child.get_text()
                    else:
                        # sorunding spaces are important !!
                        # to make sure that spacy tokenizes correctly
                        text  += " <MENTION> {} <\MENTION> ".format(child.get_text())  
        return text
    
    @staticmethod
    def check_token(token):
        """
        Takes a token and returns True if the token contains a stop-word.
        Checks is the token is one of the following:
            a stop word
            a punctuation sign
            a string of digits
            a urls/email
            a space
            has upos="NUM" (e.g. "two appels")
        """
        if token.is_stop:
            return False
        if token.is_punct:
            return False
        if token.like_num:     # replaces .is_digit + .pos=NUM
            return False
        if token.like_url:     # replaces .is_digit + .pos=NUM
            return False
        if token.like_email:   # replaces .is_digit + .pos=NUM
            return False
        if token.is_space:     # sorts out the damn \xa0 
            return False
        return True
    
    @staticmethod
    def token2text(token):
        """
        Returns the text of the given spacy token.
        Text is lower case.
        """
        return token.text.lower()
    
    def preprocess_text(self, text):
        """
        Tokenizes the text string using spacy and filters all stop-words. 
        Returns a list of lower cased tokens (str).
        """
        #
        if len(text) > 999999:
            text_parts = []
            while len(text) > 999999:
                i = 0
                while text[999999-i] != " ":
                    i += 1
                npart = text[:999999-i]
                rest = text[999999-i+1:]
                text_parts.append(npart)
                text = rest
        else:
            text_parts = [text,]
        tokens = []
        for part in text_parts:
            doc = self.nlp(part)
            tokens.extend(self.token2text(t) for t in doc if self.check_token(t))
        return tokens
    
    @staticmethod
    def find_mentions(tokens):
        """
        In the specified list of tokens (str), finds the spans of all tokens enclosed in mention markers.
        """
        # find the mention-spans in tokenized text
        begin = None
        end = None
        spans = []
        for i,t in enumerate(tokens):
            if t == "<mention>":
                begin = i+1
            elif t == "<\mention>":
                end = i
                if (begin is None) or (end is None):
                    print("DEBUG-ERROR-1")
                    return None
                spans.append((begin,end))
                begin,end=None,None
        return spans
    
    def extract_context_windows(self, tokens, spans):
        """
        Given a list of tokens and a list of spans (tupels), returns a list of dictionarys.
        These dictionarys contain the mention-texts of the given spans
        as well as the left and the right context window of the mentions.
        """
        data = []
        for begin,end in spans:
            # determine indices of mention and surrounding context
            begin_ = max(0,begin-self.window_size-1)
            end_ = min(len(tokens), end+self.window_size+1)
            # get context and mention text
            left_context = tokens[begin_:begin-1]
            right_context = tokens[end+1:end_]
            mention = tokens[begin:end]
            data.append({"mention_text":mention,
                         "left_context":left_context,
                         "right_context":right_context,})
        return data
