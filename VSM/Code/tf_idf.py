# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math

def tf_idf(corpus, test , package):
	
	dictlist = {}
	doclen = {}
	docname = package ["docname"]
	weights = package ["weights"]

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		doclen[docl] = i["length"]
		for j in i["split_sentence"]:
			# 计算 dictlist : doc —— word —— frequency
			if not dictlist.has_key(docl):
				dictlist[docl] = {}
			if not dictlist[docl].has_key(j):
				dictlist[docl][j] = 0
			dictlist[docl][j] += 1
			if test==0:
				# 计算 doclist :  word ——　doc set
				if not weights.has_key(j):
					weights[j] = set()
				weights[j].add(docl)
				docname.add(docl)
	if test ==0:
		# 计算 idf 值
		for word in weights:
			weights[word] = math.log( ( 1+len(docname)*1.0)/(len(weights[word])*1.0),2)

	# 计算 tf-idf 值
	tf_idf = {}
	for doc in dictlist:
		tf_idf[doc] = {}
		for word in dictlist[doc]:
			# tf:
			tf_idf[doc][word] = dictlist[doc][word]*1.0 / (doclen[doc]*1.0)
			# tf*idf
			tf_idf[doc][word] *= weights[word]
	package ["docname"] = docname
	package ["weights"] = weights
	#print tf_idf
	return tf_idf