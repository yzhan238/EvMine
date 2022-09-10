
from utils import *
from newlens_embs import run_newlens_emb
from newlens_match import *




kws2docID_file='../data/hkprotest/docID2keywords_all.txt'
DOC2TIME_FILE='../data/hkprotest/doc2time.txt'
OUTPUT_FILE='../data/hkprotest/test1.json'
STARTING_DATE='2019-03-30'
END_DATE='2019-12-30'
EDGE_TYPE='match'
DATASET='hkprotest'
EMBS_FILE='../data/hkprotest/emb.npy'
TOPIC_THRES=0.9
EMB_THRES=0.7
MATCH_THRES=3
TIME_INTERVAL=4
WINDOW_OVERLAP=2





docID2kws,kw2docIDs=load_keywords_info(kws2docID_file)
docID2time,time2docID=load_doc_time(DOC2TIME_FILE)
if EDGE_TYPE=='emb':
	run_newlens_emb(STARTING_DATE,docID2kws,docID2time,OUTPUT_FILE,end_date=END_DATE,embs_file=EMBS_FILE,time2docID=time2docID,match_thres=3,time_interval=5,window_overlap=3,topic_thres=TOPIC_THRES,emb_thres=EMB_THRES,edge_type=EDGE_TYPE)
elif EDGE_TYPE=='match':
	run_newlens_match(STARTING_DATE,docID2kws,docID2time,OUTPUT_FILE,end_date=END_DATE,time2docID=time2docID,match_thres=3,time_interval=5,window_overlap=3,topic_thres=TOPIC_THRES,emb_thres=EMB_THRES,edge_type=EDGE_TYPE)
elif EDGE_TYPE=='match_2':
	run_newlens_match_2(STARTING_DATE,docID2kws,docID2time,OUTPUT_FILE,end_date=END_DATE,embs_file=EMBS_FILE,time2docID=time2docID,match_thres=3,time_interval=5,window_overlap=3,topic_thres=TOPIC_THRES,emb_thres=EMB_THRES,edge_type=EDGE_TYPE)

