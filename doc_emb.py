import json
from tqdm import tqdm
import numpy as np
import datetime
import copy
import argparse
from utils import *
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os


def get_doc_emb(args, model_name = 'sentence-transformers/stsb-roberta-base-v2'):
    doc2time, _, _, _ = load_doc_time(os.path.join('data', args.data, args.doc_time))
    _, doc_sents = load_ucphrase(os.path.join('data', args.data, args.ucphrase_res), len(doc2time))

    model = SentenceTransformer(model_name).cuda()

    doc_emb = []
    for sents in tqdm(doc_sents):
        doc_emb.append(np.mean(model.encode(sents[:3]), axis=0))
    np.save(os.path.join('data', args.data, args.out), np.array(doc_emb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='hkprotest')
    parser.add_argument("--ucphrase_res", type=str, default='doc2sents-0.9-tokenized.id.json')
    parser.add_argument("--doc_time", type=str, default='doc2time.txt')
    parser.add_argument("--out", type=str, default='doc_emb')
    args = parser.parse_args()

    get_doc_emb(args)