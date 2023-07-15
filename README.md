# EvMine

The source code used for paper "[Unsupervised Key Event Detection from Massive Text Corpora](https://arxiv.org/abs/2206.04153)", published in KDD 2022.

## Requirements
Python 3 and the following packages are required: numpy, sklearn, igraph, inflect, nltk, datefinder.  
You will also need huggingface transformers package if you want to obtain document and phrase embeddings on your own corpus.

## Data

The two datasets used in the paper are available [here](https://www.dropbox.com/sh/48ezsu1eo5fii1w/AACHWV2uX-QO6D6uoJuEVpbKa?dl=0), including their original corpora, UCPhrase results, phrase embeddings, document embeddings, document publication times, and event labels (which is only used for evaluation). After downloading the dataset, put them under the **./data/** folder.  
If running on your own data, please create a dataset folder and first use [UCPhrase](https://github.com/xgeric/UCPhrase-exp) with tagging mode to mine quality phrases from the corpus. Then, you can get the phrase embeddings via

```Bash
python phrase_emb.py \
    --data hkprotest \
    --ucphrase_res doc2sents-0.9-tokenized.id.json \
    --doc_time doc2time.txt \
    --out phrase_emb
```

and document embeddings via

```Bash
python doc_emb.py \
    --data hkprotest \
    --ucphrase_res doc2sents-0.9-tokenized.id.json \
    --doc_time doc2time.txt \
    --out doc_emb
```

## Run EvMine

Use the following command to run EvMine and the results will be saved to the corresponding dataset folder.

```Bash
python EvMine.py \
    --data hkprotest \
    --ucphrase_res doc2sents-0.9-tokenized.id.json \
    --doc_time doc2time.txt \
    --doc_emb doc_emb.npy \
    --phrase_emb phrase_emb \
    --out output.json
```

## Evaluation

Use the following command to evaluate the key event detection results, where the argument eval_top refers to k for the k-Matched measure.

```Bash
python eval.py \
    --key_event_file data/hkprotest/output.json \
    --ground_truth data/hkprotest/doc2event_id.txt \
    --eval_top 5
```

## Citations

If you find our work useful for your research, please cite the following paper:
```
@inproceedings{Zhang2022EvMine,
  title={Unsupervised Key Event Detection from Massive Text Corpora},
  author={Yunyi Zhang and Fang Guo and Jiaming Shen and Jiawei Han},
  booktitle={KDD},
  year={2022}
}
```
