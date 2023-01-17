"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
import enum
import os
from collections import defaultdict, Counter
from pdb import post_mortem
import dill
from copyreg import dispatch_table
from dill import Pickler
from numpy import format_float_scientific;dispatch_table
import pickle
import math
import operator
import code
from sre_parse import Tokenizer
import nltk

from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

#dbfile= "index.pkl"
nltk.download('wordnet')
nltk.download('omw-1.4')
dbfile =  os.environ["HOMEPATH"] + "\OneDrive\Desktop\PA2\EntryInfo.pickle"

class Indexer:
      # You need to store index file on your disk so that you don't need to
                         # re-index when running the program again.
                         # You can name this file however you like. (e.g., index.pkl)



    def __init__(self):
        # TODO. You will need to create appropriate data structures for the following elements
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))   # map (token to id)
        self.idx2tok = dict()                       # map (id to token)
        self.postings_lists = dict()                # postings for each word
        self.docs = []                            # encoded document list            
        self.corpus_stats = { 'avgdl': 0 }        # any corpus-level statistics
        self.stopwords = stopwords.words('english')
        self.entry = {'tok2idx', 'idx2tok', 'postings_lists', 'docs', 'raw_ds', 'corpus_stats', 'stopwords'}
        if os.path.exists(dbfile):
            with open(dbfile, 'rb') as handle:
                self.entry = dill.load(handle)
            self.tok2idx = self.entry['tok2idx']
            self.idx2tok = self.entry['idx2tok']
            self.postings_lists = self.entry['postings_lists']
            self.docs = self.entry['docs']
            self.raw_ds = self.entry['raw_ds']
            self.corpus_stats = self.entry['corpus_stats']
            self.stopwords = self.entry['stopwords']
            
        else:
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds, False)
            self.create_postings_lists()
            with open(dbfile, 'wb') as handle:
                dill.dump(self.entry, handle, protocol= pickle.HIGHEST_PROTOCOL)
                



    def clean_text(self, lst_text, query=False):
        # TODO. this function will run in two modes: indexing and query mode.
        # TODO. run simple whitespace-based tokenizer (e.g., RegexpTokenizer)
        # TODO. run lemmatizer (e.g., WordNetLemmatizer)
        # TODO. read documents one by one and process
        tokenizer = RegexpTokenizer(r"\w+")
        lemmatizer = WordNetLemmatizer()
        for l in tqdm(lst_text):
            enc_doc = []
            query_doc = []
            #lowercase, remove extra whitespaces
            l = l.lower().strip()
            #tokenize
            seq = tokenizer.tokenize(l)
            #remove stopwords
            seq = [w for w in seq if w not in self.stopwords]
            seq = [lemmatizer.lemmatize(w) for w in seq]
            if query is False:
                for w in seq:
                    self.idx2tok[self.tok2idx[w]] = w
                    enc_doc.append(self.tok2idx[w])
                self.docs.append(enc_doc)
            else:
                for w in seq:
                    query_doc.append(w)
        
        if query is True:
            return query_doc

    
    

        

    def create_postings_lists(self):
        # TODO. This creates postings lists of your corpus
        # TODO. While indexing compute avgdl and document frequencies of your vocabulary
        # TODO. Save it, so you don't need to do this again in the next runs.
        # Save
        for docid, d in enumerate(self.docs):
            self.corpus_stats['avgdl'] += len(d)
            #print (str(d))
            for i in d:
                #print (str(docid))
                if i in self.postings_lists:
                    self.postings_lists[i][0] += 1 #increment document frequency
                    self.postings_lists[i][1].append(docid)
            
                else: 
                    self.postings_lists[i] = [1, [docid]]
        self.corpus_stats['avgdl'] /= len(self.docs)
        self.entry = {
            'tok2idx' : self.tok2idx,
            'idx2tok' : self.idx2tok,
            'postings_lists' : self.postings_lists,
            'docs' : self.docs,
            'raw_ds' : self.raw_ds,
            'corpus_stats' : self.corpus_stats,
            'stopwords' : self.stopwords
        }
    



class SearchAgent:
    k1 = 1.5                # BM25 parameter k1 for tf saturation
    b = 0.75                # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        # TODO. set necessary parameters
        self.i = indexer
        self.avgl = float(i.corpus_stats['avgdl'])



    def query(self, q_str):
        # TODO. This is take a query string from a user, run the same clean_text process,
        q_indx = self.i.clean_text([q_str], query = True)
        #code.interact(local=dict(globals(), **locals()))
        results = {}
        for t in q_indx:
            df = self.i.postings_lists[q.i.tok2idx[t]][0]
            w = math.log2((len(self.i.docs) - df + 0.5) / (df + 0.5))
            tf = 0
            for docid in self.i.postings_lists[q.i.tok2idx[t]][1]:
                tf = self.i.postings_lists[q.i.tok2idx[t]][1].count(docid)
                dl = len(self.i.docs[docid])
                s = (self.k1 * tf * w) / (tf + self.k1 * (1 - self.b + self.b * dl/ self.avgl))
                if docid in results:
                    results[docid] += s
                else:
                    results[docid] = s
        results = sorted(results.items(), key = operator.itemgetter(1))
        results.reverse()

        
        # TODO. Calculate BM25 scores
        # TODO. Sort  the results by the scores in decsending order
        # TODO. Display the result

        
        if len(results) == 0:
            return None
        else:
            self.display_results(results)


    def display_results(self, results):
        # Decode
        # TODO, the following is an example code, you can change however you would like.
        for docid, scores in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {scores}')
            print('Article:')
            print(self.i.raw_ds[docid])



if __name__ == "__main__":
    i = Indexer()           # instantiate an indexer
    q = SearchAgent(i)      # document retriever
    code.interact(local=dict(globals(), **locals())) # interactive shell