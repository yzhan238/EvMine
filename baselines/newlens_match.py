from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import argparse
from nltk import sent_tokenize
from utils import *
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import igraph as ig
from math import sqrt
import copy
from numpy import dot
from numpy.linalg import norm
import datetime
import json


def generate_topics(docIDs,docID2kws,docID2time,edge_type='match',match_thres=3,pretrained_embs=[],emb_thres=0.3):
##############################################
##OUTPUT: 
##1\match: [sorted_times,clus_docIDs,kw_counter]
##2\embeds: [sorted_times,clus_docIDs,avg_embs]
#########################################
	mat = np.zeros((len(docIDs), len(docIDs)))
	#edge type 1: building matrix using match
	docIDs=list(docIDs)
	edges = []
	weights = []
	if edge_type=='match':
		for i,docID1 in enumerate(docIDs):
			for j,docID2 in enumerate(docIDs):
				if i >=j:
					continue
				kws1=docID2kws[docID1]
				kws2=docID2kws[docID2]
				intersect=set(kws1) & set(kws2)
				if len(intersect)>= match_thres:
					#print(intersect,docID1,docID2)
					#print()
					mat[i,j]=1
					edges.append((i,j))
					weights.append(mat[i, j])
		print('NUM of edges is ',len(edges))
		g = ig.Graph()
		g.add_vertices(len(docIDs))
		g.add_edges(edges)
		levels = g.community_multilevel(weights=weights, return_levels=True)
		topics=[]
		for ci, c in enumerate(levels[0]):
			kw_counter=defaultdict(int)
			for idx in c:
				for kw in docID2kws[docIDs[idx]]:
					kw_counter[kw]+=1
			clus_docIDs=set([docIDs[idx] for idx in c])
			times=set([docID2time[docID] for docID in clus_docIDs])
			sorted_times=sorted(times)
			topics.append((sorted_times,clus_docIDs,kw_counter))
			
		
		return topics
	#edge type 1: building matrix using embeds similarity
	if edge_type=='embs':
		for i,docID1 in enumerate(docIDs):
			for j,docID2 in enumerate(docIDs):
				if i >=j:
					continue
				emb1=pretrained_embs[docID1]
				emb2=pretrained_embs[docID2]
				cos_sim = dot(emb1, emb2)/(norm(emb1)*norm(emb2))
				#print(emb1,emb2,cos_sim)
				#sys.exit(1)
				if cos_sim>= emb_thres:
					#print(intersect,docID1,docID2)
					#print()
					mat[i,j]=1
					edges.append((i,j))
					weights.append(mat[i, j])
		g = ig.Graph()
		g.add_vertices(len(docIDs))
		g.add_edges(edges)
		levels = g.community_multilevel(weights=weights, return_levels=True)
		topics=[]
		for ci, c in enumerate(levels[0]):
			kw_counter=defaultdict(int)
			total_embs=[]
			for idx in c:
				total_embs.append(pretrained_embs[docIDs[idx]])
			total_embs=np.array(total_embs)
			#print(total_embs.shape)
			avg_emb=np.mean(total_embs, axis=0)
			clus_docIDs=set([docIDs[idx] for idx in c])
			#print(avg_emb.shape)
			#clus_info: (sorted_time,docIDs,kw_counter)
			#clus_docIDs=set([docIDs[idx] for idx in c])
			times=set([docID2time[docID] for docID in clus_docIDs])
			sorted_times=sorted(times)
			topics.append((sorted_times,clus_docIDs,avg_emb))
		return topics

#using louvain clustering to do the merging
def topics_matching_louvain(topic_keeper,new_topics,docID2kws,cur_date,docID2time,time_interval=3,topic_thres=0.6,min_doc_cnt=3):
	merges1=[]
	merges2=[]
	merge_keeper=defaultdict(list)
	mergee_record=set()
	merger_record=set()
	dups=set()
	adds=[]
	mat = np.zeros((len(topic_keeper)+len(new_topics), len(topic_keeper)+len(new_topics)))
	edges = []
	weights = []
	for id1,topic1 in enumerate(topic_keeper):
		for id2, topic2 in enumerate(new_topics):
			#calculate similarity:
			if topic1[1]==topic2[1]:
				continue
			if topic2[1].issubset(topic1[1]):
				print('SKIP subset',topic1[1],topic2[1])
				continue
			intersect=set(topic1[2].keys())&set(topic2[2].keys())
			A=0
			for kw in intersect:
				A+=topic1[2][kw]*topic2[2][kw]
			if A==0:
				score=0
			else:
				B=0
				for kw,cnt in topic1[2].items():
					B+=cnt**2
				B=sqrt(B)
				C=0
				for kw,cnt in topic2[2].items():
					C+=cnt**2
				C=sqrt(C)
				score=A/(B*C)
				if score > topic_thres:
					
					latest_time=topic1[0][-1]
					d1 = datetime.datetime.strptime(cur_date, "%Y-%m-%d")
					d2 = datetime.datetime.strptime(latest_time, "%Y-%m-%d")
					interval=str(d1-d2)
					if interval!='0:00:00' and int(interval.split(', ')[0].split(' ')[0])>=2*time_interval:
						print('SKIP OVERTIME MERGE')
						continue
					#recording merging topics
					print('merging topics!!!!!!!')
					print(topic1[1],topic2[1])
					print(A,B,C,score)
					print()
					merge_keeper[id2].append(id1)
					mergee_record.add(id1)
					merger_record.add(id2)
					
					mat[id1,len(topic_keeper)+id2]=1
					mat[len(topic_keeper)+id2,id1]=1
					edges.append((id1,len(topic_keeper)+id2))
					edges.append((len(topic_keeper)+id2,id1))
					weights.append(mat[id1, len(topic_keeper)+id2])
					weights.append(mat[len(topic_keeper)+id2, id1])
					
	#for (id1,id2) in zip(merges1,merges2):
	new_topic_keeper=[]
	new_docIDs_record=set()
	# Copy from previous topics
	for id1,topic1 in enumerate(topic_keeper):
		if id1 in mergee_record:
			print('###SKIP ALREADY MERGED1 :',topic1[1])
			#print()
			continue
		latest_time=topic1[0][-1]
		d1 = datetime.datetime.strptime(cur_date, "%Y-%m-%d")
		d2 = datetime.datetime.strptime(latest_time, "%Y-%m-%d")
		interval=str(d1-d2)
		docIDs=topic1[1]
		if len(docIDs) < min_doc_cnt and interval!='0:00:00' and int(interval.split(', ')[0].split(' ')[0])>2*time_interval:
			print('###SKIP TIME OVERDUE :',topic1[1])
			#print(topic1)
			continue
		if docIDs not in new_docIDs_record:
			new_docIDs_record.add(frozenset(docIDs))
			new_topic_keeper.append(copy.deepcopy(topic1))
			
	# Copy from new merges
	#  
	g = ig.Graph()
	g.add_vertices(len(topic_keeper)+len(new_topics))
	g.add_edges(edges)
	components = g.clusters()
	for ci, c in enumerate(components):
		kw_counter=defaultdict(int)
		times=set()
		clus_docIDs=set()
		if len(c) <2:
			continue
		for idx in c:
			if idx < len(topic_keeper):
				topic=topic_keeper[idx]
			else:
				topic=new_topics[idx-len(topic_keeper)]
			for docID in topic[1]:
				for kw in docID2kws[docID]:
					kw_counter[kw]+=1
				times.add(docID2time[docID])
				clus_docIDs.add(docID)
		#clus_info: (sorted_time,docIDs,kw_counter)
		#times=set([docID2time[docID] for docID in clus_docIDs])
		sorted_times=sorted(times)
		print('MERGING RESULT :',clus_docIDs)
		if clus_docIDs not in new_docIDs_record:
			new_docIDs_record.add(frozenset(clus_docIDs))
			new_topic_keeper.append((sorted_times,clus_docIDs,kw_counter))
	
	# Copy from new topics
	#
	#
	for id2, topic2 in enumerate(new_topics):
		if id2 in merger_record:
			print('###SKIP ALREADY MERGED2 :',topic2[1])
			continue
		docIDs=topic2[1]
		if docIDs not in new_docIDs_record:
			#print('###ADD NEW TOPIC',docIDs)
			new_docIDs_record.add(frozenset(docIDs))
			new_topic_keeper.append(copy.deepcopy(topic2))
			
	return new_topic_keeper



def rolling_topics(init_date,docID2kws,docID2time,match_thres,time_interval,window_overlap,
				   topic_thres,end_date,edge_type,time2docID,emb_thres=0.3,embs_file=''):
	init_docIDs=find_docids_by_time(init_date,time2docID,interval=time_interval)
	init_topics=generate_topics(init_docIDs,docID2kws,docID2time,edge_type=edge_type,pretrained_embs=[],
								emb_thres=emb_thres,match_thres=match_thres)
	topic_keeper=init_topics
	for topic in topic_keeper:
		if len(topic[1])>=2:
			print(topic[1])
	init_date = datetime.datetime.strptime(init_date, "%Y-%m-%d")
	end_date= datetime.datetime.strptime(end_date, "%Y-%m-%d")
	cur_date = init_date + datetime.timedelta(days=window_overlap)
	days_to_end=str(end_date-cur_date)
	cur_date=str(cur_date).split(' ')[0]
	cur_docIDs=find_docids_by_time(cur_date,time2docID,interval=time_interval)
	
	cnt=0
	#MOD: can not use # of current docIDs as condition
	while days_to_end!='0:00:00' and int(days_to_end.split(', ')[0].split(' ')[0])>0:
		#print('#####################################')
		print('DAYS TO END :',int(days_to_end.split(', ')[0].split(' ')[0]))
		print(cur_date,datetime.datetime.strptime(cur_date, "%Y-%m-%d") + datetime.timedelta(days=time_interval))
		try:
			cur_topics=generate_topics(cur_docIDs,docID2kws,docID2time,edge_type=edge_type,
								emb_thres=emb_thres,match_thres=match_thres)
			#print(cur_topics)
			
		except:
			cur_topics=[]
		print(len(topic_keeper),len(cur_topics))
		if edge_type=='match':
			topic_keeper=topics_matching_louvain(topic_keeper,cur_topics,docID2kws,cur_date,docID2time,time_interval=time_interval,topic_thres=topic_thres)
		print('####CURRENT TOPIC KEEPER:')
		for topic in topic_keeper:
			if len(topic[1])>=2:
				print(topic[1])
		print()
		
		cur_date=datetime.datetime.strptime(cur_date, "%Y-%m-%d")+datetime.timedelta(days=window_overlap)
		days_to_end=str(end_date-cur_date)
		cur_date=str(cur_date).split(' ')[0]
		cur_docIDs=find_docids_by_time(cur_date,time2docID,interval=time_interval)
		cnt+=1
		#if cnt==1:
			#break
	 
	#print(topics)
	return topic_keeper


def sort_topic_matching_edges(cluster,docID2kws,match_thres):
	edge_counter=defaultdict(int)
	for i,docID1 in enumerate(cluster):
		for j,docID2 in enumerate(cluster):
			if i >=j:
				continue
			kws1=docID2kws[docID1]
			kws2=docID2kws[docID2]
			intersect=set(kws1) & set(kws2)
			if len(intersect)>= match_thres:
				edge_counter[docID1]+=1
				edge_counter[docID2]+=1
	new_cluster=[(docID,edge_counter[docID])for docID in cluster]
	sorted_cluster=sorted(new_cluster, key=lambda tup: -tup[1] )
	#print(sorted_cluster)
	sorted_cluster=[tp[0] for tp in sorted_cluster]
	#print(sorted_cluster)
	return sorted_cluster



def run_newlens_match(STARTING_DATE,docID2kws,docID2time,OUTPUT_FILE,end_date,time2docID,match_thres=3,time_interval=5,window_overlap=3,
							topic_thres=0.9,emb_thres=0.7,edge_type='match'):

	topic_keeper=rolling_topics(STARTING_DATE,docID2kws,docID2time,match_thres=3,time_interval=5,window_overlap=3,
							topic_thres=topic_thres,emb_thres=emb_thres,end_date=end_date,edge_type=edge_type,time2docID=time2docID)
	with open(OUTPUT_FILE, 'w') as f:
		output=[]
		for topic in topic_keeper:
			if len(topic[1])>=5:
				cluster=[docID for docID in topic[1]]
				sorted_cluster=sort_topic_matching_edges(cluster,docID2kws=docID2kws,match_thres=match_thres)
				output.append(sorted_cluster)
		json.dump(output, f, indent=2)


