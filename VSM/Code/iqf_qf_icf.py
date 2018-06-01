# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np

def iqf_qf_icf(corpus,test,package):
	"""
	labell : 目录名
	dictlist: 目录名——文档——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
	doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
	numdoc : 每个label包含的文档名的集合，用于获取该label下文档的数目
	worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）
	"""
	weights = package["weights"]
	labelset = package["labelset"]

	doclist = {}
	dictlist = {}
	worddict = {}
	word_label_dict = {}
	n = len(corpus)

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		if not doclist.has_key(labell):
			doclist[labell] = {}
		if not dictlist.has_key(labell):
			dictlist[labell] = {}
		if not dictlist[labell].has_key(docl):
			dictlist[labell][docl] = {}	
		for j in i["split_sentence"]:
			# 计算 dictlist : label —— doc —— word —— frequency
			if not dictlist[labell][docl].has_key(j):
				dictlist[labell][docl][j] = 1
			else:
				dictlist[labell][docl][j] += 1
			# 计算 doclist : label —— word ——　doc set
			if not doclist[labell].has_key(j):
				doclist[labell][j] = set()
			doclist[labell][j].add(docl)

	iqf_qf_icf = {}

	if test == 1:
		for label in doclist:
			iqf_qf_icf[label] = {}
			for word in doclist[label]:
				if weights[label].has_key(word):
					iqf_qf_icf[label][word] = weights[label][word]
				else:
					iqf_qf_icf[label][word] = max([weights[x][word] for x in weights if weights[x].has_key(word)])
		return iqf_qf_icf

	# 获取每个词出现的文档数目（不看labell）
	# a = len(docllist[labell][word])
	# b = word出现的文档数目 - a
	# c = 用for求sum(doclist[label][!word])
	# d = sum( !word 出现的文档数目 ) - c
	# iqf_qf_icf = totaldoc * sqrt(ad-bc) / ( (a+b)*(a+c)*(d+b)*(c+d)  )

	
	for labell in labelset:
		iqf_qf_icf[labell] = {}
		for word in doclist[labell]:
			if not word_label_dict.has_key(word):
				word_label_dict[word] = set()
			word_label_dict[word].add(labell)
			if not worddict.has_key(word):
				worddict[word] = 0
			worddict[word] += len(doclist[labell][word])

	# 计算 iqf_qf_icf
	for labell in labelset:
		weights[labell] = {}
		for word in doclist[labell]:
			a_b = worddict[word]
			a = len(doclist[labell][word])
			b = a_b - a
			weights[labell][word] = math.log(n*1.0/(a+b),2)*math.log(a+1,2)*math.log( len(labelset)*1.0/len(word_label_dict[word]) +1,2)
	
	package["weights"] = weights
	package["labelset"] = labelset
	return weights
