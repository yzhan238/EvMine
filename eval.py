from collections import Counter
import os
import json
import argparse


def eval(args):
    key_events = json.load(open(args.key_event_file))
    thres = args.eval_top // 2 + 1

    gt = {}
    all_ev_id = set()
    with open(args.ground_truth) as f:
        for line in f:
            doc_id, ev_id = line.strip().split('\t')
            if ev_id != 'X':
                gt[int(doc_id)] = ev_id
                all_ev_id.add(ev_id)

    tp = set()
    p = 0
    for key_event in key_events:
        if len(key_event) < args.eval_top:
            continue
        p += 1
        c = Counter([gt[did] for did in key_event[:args.eval_top] if did in gt])
        for gt_ev, count in c.items():
            if count >= thres:
                tp.add(gt_ev)
                break
    prec = len(tp) / float(p)
    print(f'{args.eval_top}-Precision: {prec}')
    recall = len(tp) / float(len(all_ev_id))
    print(f'{args.eval_top}-Recall: {recall}')
    f1 = 2 * prec * recall / (prec + recall)
    print(f'{args.eval_top}-F1: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_event_file", type=str)
    parser.add_argument("--ground_truth", type=str)
    parser.add_argument("--eval_top", type=int)
    args = parser.parse_args()

    eval(args)