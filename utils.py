import datetime
import copy
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cos
from collections import defaultdict
import os


def load_doc_time(file):
    doc2time = []
    with open(file) as f:
        for i, d2t in enumerate(f):
            doc_id, t = d2t.strip().split('\t')
            assert int(doc_id) == i
            t = datetime.datetime.strptime(t, '%Y-%m-%d')
            doc2time.append(t)
    min_t = min(doc2time)
    num_t = (max(doc2time)-min(doc2time)).days + 1
    all_t = [min(doc2time)+datetime.timedelta(days=i) for i in range(num_t)]
    return doc2time, min_t, num_t, all_t


def load_ucphrase(file, num_doc):
    data=json.load(open(file))
    doc_sents = []
    docs = []
    for doc_id in range(num_doc):
        doc = data[str(doc_id)]
        char = doc[0]['tokens'][0][0]
        sents = []
        for sent in doc:
            tokens = copy.deepcopy(sent['tokens'])
            for s, e, p in sent['spans']:
                rep = ['' for _ in range(s, e+1)]
                rep[0] = ' ' + p.replace(' ', '_')
                tokens[s:e+1] = rep
            sents.append(''.join(tokens).replace(char, ' ').strip())
        doc_sents.append(sents)
        docs.append(' '.join(sents))
    return docs, doc_sents


def get_phrase_emb_sim(args):
    (p_emb, i2p, p2i) = pickle.load(open(os.path.join('data', args.data, args.phrase_emb+'.pkl'), 'rb'))

    return cos(p_emb), i2p, p2i


def find_all(s, sub):
    start = 0
    s = ' ' + s + ' '
    sub = ' ' + sub + ' '
    while True:
        start = s.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def word_counting(w, docs):
    ret = {}
    for did, doc in enumerate(docs):
        num_matches = len(list(find_all(doc, w)))
        if num_matches > 0:
            ret[did] = num_matches
    return ret


def tf_itf(w, t, w2tc, num_t, window_size=5):
    tf = 0.
    for i in range(window_size):
        new_t = t+datetime.timedelta(days=i)
        if new_t in w2tc[w] and w2tc[w][new_t] != 0:
            tf += w2tc[w][new_t] * ((window_size-i)/float(window_size))
        elif i == 0:
            return 0, 0, 0
    itf = float(num_t) / len(w2tc[w])
    return tf/window_size * np.log(itf), tf, itf