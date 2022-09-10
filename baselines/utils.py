from nltk import sent_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import numpy as np
import datetime

def read_stopwords(stopwords_file):
    stopwords=set()
    with open(stopwords_file,'r') as In:
        for line in In:
            line=line.strip()
            stopwords.add(line)
    return stopwords


#tfidf strategy: 1\ top5 sents vs top5sents; 2\ whole docs vs whole docs;
def make_tfidf_corpus(idx,orig_corpus):
    new_corpus=[]
    #current_news=(orig_corpus[idx])
    #sents=sent_tokenize(current_news)
    new_corpus.append(orig_corpus[idx])
    #new_corpus.append(' '.join(sents[0:5]))
    new_corpus.extend(orig_corpus[0:idx])
    new_corpus.extend(orig_corpus[idx+1:])
    return new_corpus

def load_doc_time(DOC2TIME_FILE):
    docID2time=defaultdict(str)
    time2docID=defaultdict(list)
    with open(DOC2TIME_FILE) as IN:
        for line in IN:
            line=line.strip().split('\t')
            docid=int(line[0])
            time=line[1]
            #print(doc_id,time)
            time2docID[time].append(docid)
            docID2time[docid]=time
    #print(docID2time[1],time2docID['2019-08-08'])
    return docID2time,time2docID


def output_docID2keywords(docID2keywords,OUT):
    with open(OUT,'w+') as OUT:
        for ks in docID2keywords:
            temp=[]
            for k in ks:
                temp.append(k[0]+':'+str(k[1]))
            OUT.write(' '.join(temp)+'\n')


def load_keywords_info(keywords2docID_file,thres=0.1):
    docID2kws=defaultdict(list)
    with open(keywords2docID_file) as IN:
        for docID,line in enumerate(IN):
            if line=='\n':
                docID2kws[docID]=[]
                continue
            line=line.strip().split(' ')
            for tp in line:
                tp=tp.split(':')
                kw=tp[0]
                score=float(tp[1])
                #print(tp)
                if score>thres:
                    docID2kws[docID].append(kw)
    kw2docIDs=defaultdict(list)
    for docID,kws in docID2kws.items():
        for kw in kws:
            kw2docIDs[kw].append(docID)
    return docID2kws,kw2docIDs


def load_keywords_info_miranda(keywords2docID_file):
    docID2kwDic=defaultdict(list)
    with open(keywords2docID_file) as IN:
        for docID,line in enumerate(IN):
            if line=='\n':
                docID2kwDic[docID]={}
                continue
            line=line.strip().split(' ')
            kw_dic={}
            for tp in line:
                tp=tp.split(':')
                kw=tp[0]
                score=float(tp[1])
                kw_dic[kw]=score
            docID2kwDic[docID]=kw_dic
    return docID2kwDic


def load_embs_data(IN):
    data=np.load(IN)
    return data

def tfidf(corpus,vocab_size,stopwords,thres=0.1):

    vectorizer=CountVectorizer()
    tf = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words=stopwords)

    transformer=TfidfTransformer()
    tfidf=tf.fit_transform(corpus)
    word = tf.get_feature_names() 
    #word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    c=[]
    for j in range(len(word)):
        c.append((word[j],weight[0][j]))
    
    c.sort(key=lambda tup: tup[1],reverse=True)
    word_list=[]
    #print(c)
    for tp in c:
        if tp[1]<thres:
            break
        word_list.append(tp)
        '''
        if '_' in tp[0] or tp[0] in SINGLES:
            word_list.append(tp)
        '''
    #new_vocab=set(word_list)
    return word_list



def find_docids_by_time(start_date,time2docID,interval=2):
    docids=set()
    date_1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    for i in range(interval):
        end_date = date_1 + datetime.timedelta(days=i)
        end_date=str(end_date).split(' ')[0]
        #print(time2docID[end_date])
        for docid in time2docID[end_date]:
            docids.add(docid)
    return docids


