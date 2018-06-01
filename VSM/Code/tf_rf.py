# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np


def tf_rf(corpus,test,package):
	"""
	dictlist: 目录名——文档——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
	doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
	worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）
	"""
	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]

	doclen = {}
	dictlist = {}
	worddict = {}
	totaldoc = len(corpus)

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		doclen[docl] = i["length"]
		if not doclen.has_key(labell):
			doclen[labell] = {}
		if not doclen[labell].has_key(docl):
			doclen[labell][docl] = 0
		doclen[labell][docl] += i["length"]

		for j in i["split_sentence"]:
			# 计算 dictlist : label —— doc —— word —— frequency
			if not dictlist.has_key(labell):
				dictlist[labell] = {}
			if not dictlist[labell].has_key(docl):
				dictlist[labell][docl] = {}
			if not dictlist[labell][docl].has_key(j):
				dictlist[labell][docl][j] = 0
			dictlist[labell][docl][j] += 1
			if test ==0 :
				# 计算 doclist : label —— word ——　doc set
				if not doclist.has_key(labell):
					doclist[labell] = {}
				if not doclist[labell].has_key(j):
					doclist[labell][j] = set()
				doclist[labell][j].add(docl)

		# 获取每个词出现的文档数目（不看labell）
		# a = len(docllist[labell][word])
		# c = 用for求sum(doclist[label][!word])
		# rf = a/c
	if test ==0:
		for labell in labelset:
			weights[labell] = {}
			for word in doclist[labell]:
				if not worddict.has_key(word):
					worddict[word] = 0
				worddict[word] += len(doclist[labell][word])
				a = len(doclist[labell][word])
				c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
				weights[labell][word] = math.log(2+a*1.0/max(1,c*1.0),2)

	# 计算 tf-rf 值
	tf_rf = {}
	for labell in labelset:
		tf_rf[labell] = {}
		for doc in dictlist[labell]:
			tf_rf[labell][doc] = {}
			for word in dictlist[labell][doc]:
				#print doc + word
				tf_rf[labell][doc][word] = dictlist[labell][doc][word]*1.0 / (doclen[labell][doc]*1.0)
				if weights[labell].has_key(word):
					tf_rf[labell][doc][word] *= weights[labell][word]
				else:
					tf_rf[labell][doc][word] *= max([ weights[x][word]  for x in weights if weights[x].has_key(word)])
	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist
	return tf_rf
