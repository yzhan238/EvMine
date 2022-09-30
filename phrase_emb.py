import json
from tqdm import tqdm
from collections import defaultdict as ddict
import numpy as np
import datetime
import copy
import argparse
from utils import *
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pickle
import os

MAX_SENT_LEN = 300
MIN_SENT_LEN = 10

def get_sentences(doc_sents, tokenizer):
    sentences = []
    masked_sentences = []
    emb_index = []
    for di, doc in tqdm(enumerate(doc_sents)):
        for si, sent in enumerate(doc):
            phrase_tokens = sent.split(' ')
            if len(phrase_tokens) > MAX_SENT_LEN or len(phrase_tokens) < MIN_SENT_LEN:
                continue
            for ti, tok in enumerate(phrase_tokens):
                if '_' not in tok:
                    continue
                center_ids = tokenizer.encode(tok.replace('_', ' '), add_special_tokens=False)
                if len(center_ids) < 2: continue
                if ti > 0:
                    left_ids = tokenizer.encode(' '.join(phrase_tokens[:ti]).replace('_', ' '), add_special_tokens=False)
                else:
                    left_ids = []
                if ti < len(phrase_tokens) - 1:
                    right_ids = tokenizer.encode(' '.join(phrase_tokens[ti+1:]).replace('_', ' '), add_special_tokens=False)
                else:
                    right_ids = []
                emb_index.append((di, si, tok))
                sentences.append(([tokenizer.cls_token_id] + left_ids + center_ids + right_ids + [tokenizer.sep_token_id], len(left_ids) + 1, len(left_ids) + len(center_ids) + 1))
                masked_sentences.append(([tokenizer.cls_token_id] + left_ids + [tokenizer.mask_token_id] + right_ids + [tokenizer.sep_token_id], len(left_ids) + 1, len(left_ids) + 2))
    return sentences, masked_sentences, emb_index


def get_pretrained_emb(model, tokenizer, sentences, np_file, dim=768, batch_size=128):
    fp = np.memmap(np_file, dtype='float32', mode='w+', shape=(len(sentences), dim))
    iterations = int(len(sentences)/batch_size) + (0 if len(sentences) % batch_size == 0 else 1)
    this_idx = 0
    for i in tqdm(range(iterations)):
        start = i * batch_size
        end = min((i+1)*batch_size, len(sentences))
        batch_ids = [ids for ids,_,_ in sentences[start:end]]
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = model(ids, masks)[0]
        for final_layer, (_,s,e) in zip(batch_final_layer, sentences[start:end]):
            rep = np.mean(final_layer[s:e].cpu().numpy(), axis=0)
            fp[this_idx] = rep.astype(np.float32)
            this_idx += 1
    del fp
    

def get_phrase_emb(args, model_name = 'bert-base-uncased'):
    doc2time, _, _, _ = load_doc_time(os.path.join('data', args.data, args.doc_time))
    _, doc_sents = load_ucphrase(os.path.join('data', args.data, args.ucphrase_res), len(doc2time))

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = False)

    model = BertModel.from_pretrained(model_name)
    model.to(torch.device("cuda"))
    model.eval()

    sentences, masked_sentences, emb_index = get_sentences(doc_sents, tokenizer)

    get_pretrained_emb(model, tokenizer, sentences, os.path.join('data', args.data, args.out+'_unmasked.npy'))
    get_pretrained_emb(model, tokenizer, masked_sentences, os.path.join('data', args.data, args.out+'_masked.npy'))

    
    phrase_emb_masked = np.memmap(os.path.join('data', args.data, args.out+'_masked.npy'), dtype='float32', mode='r', shape=(len(emb_index), 768))
    phrase_emb_unmasked = np.memmap(os.path.join('data', args.data, args.out+'_unmasked.npy'), dtype='float32', mode='r', shape=(len(emb_index), 768))

    p2embs = defaultdict(list)
    for emb_idx, (doc_id, _, p) in enumerate(emb_index):
        p2embs[p].append(np.concatenate((phrase_emb_masked[emb_idx], phrase_emb_unmasked[emb_idx]), axis=0))
    p_emb = np.empty((len([p for p, embs in p2embs.items() if len(embs) > 2]), 768*2))
    i2p = []
    for p, embs in p2embs.items():
        if len(embs) <= 2: continue
        idx = len(i2p)
        i2p.append(p)
        p_emb[idx] = np.mean(embs, axis=0)
    p2i = {p:i for i,p in enumerate(i2p)}
    
    pickle.dump((p_emb, i2p, p2i), open(os.path.join('data', args.data, args.out+'.pkl'), 'wb'))
    
    os.remove(os.path.join('data', args.data, args.out+'_masked.npy'))
    os.remove(os.path.join('data', args.data, args.out+'_unmasked.npy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='hkprotest')
    parser.add_argument("--ucphrase_res", type=str, default='doc2sents-0.9-tokenized.id.json')
    parser.add_argument("--doc_time", type=str, default='doc2time.txt')
    parser.add_argument("--out", type=str, default='phrase_emb')
    args = parser.parse_args()

    get_phrase_emb(args)