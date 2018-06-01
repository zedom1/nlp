# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np

def tf_eccd(corpus,test,package):

	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]

	dictlist = {}
	doclen = {}
	n = len(corpus)
	worddict = {}

	# labell : 目录名
	# label: 目录列表
	# dictlist: 目录名——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
	# doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
	# worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		if not doclen.has_key(labell):
			doclen[labell] = 0
		doclen[labell] += i["length"]
		for j in i["split_sentence"]:
			# 计算 dictlist : label —— doc —— word —— frequency
			if not dictlist.has_key(labell):
				dictlist[labell] = {}
			if not dictlist[labell].has_key(j):
				dictlist[labell][j] = 1
			else:
				dictlist[labell][j] += 1

			if test ==0 :
				# 计算 doclist : label —— word ——　doc set
				if not doclist.has_key(labell):
					doclist[labell] = {}
				if not doclist[labell].has_key(j):
					doclist[labell][j] = set()
				doclist[labell][j].add(docl)


	# 获取每个词出现的文档数目（不看labell）
	# a = len(docllist[labell][word])
	# b = word出现的文档数目 - a
	# c = 用for求sum(doclist[label][!word])
	# d = sum( !word 出现的文档数目 ) - c
	# eccd = totaldoc * sqrt(ad-bc) / ( (a+b)*(a+c)*(d+b)*(c+d)  )
	if test == 0 :
		entropy = {}
		tf = {}
		for labell in labelset:
			tf[labell] = {}
			for word in doclist[labell]:
				if not worddict.has_key(word):
					worddict[word] = 0
				worddict[word] += len(doclist[labell][word])
				total = sum([dictlist[x][word]  for x in dictlist if dictlist[x].has_key(word)])
				tf[labell][word] = dictlist[labell][word]*1.0/total

		Emax = -999
		for labell in labelset:
			for word in doclist[labell]:
				if not entropy.has_key(word):
					entropy[word] = -sum([ tf[x][word]*1.0*math.log(tf[x][word],2) for x in labelset if tf[x].has_key(word) ])
					Emax = max(entropy[word],Emax)

		# 计算 eccd
		for labell in labelset:
			weights[labell] = {}
			for word in doclist[labell]:
				a_b = worddict[word]
				a = len(doclist[labell][word])
				b = a_b - a
				c_d = sum([(worddict[x]) for x in worddict.keys() if x!=word])
				c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
				d = c_d - c
				weights[labell][word] = ((a*d-b*c)*1.0/((a+c)*(b+d)))* (Emax - entropy[word])*1.0/Emax

	#print eccd
	# 计算 tf-eccd 值
	tf_eccd = {}
	for labell in labelset:
		tf_eccd[labell] = {}
		for word in dictlist[labell]:
			tf_eccd[labell][word] = dictlist[labell][word]*1.0 /  (doclen[labell]*1.0)
			if weights[labell].has_key(word):
				tf_eccd[labell][word] *= weights[labell][word]
			else:
				tf_eccd[labell][word] *= max([ weights[x][word]  for x in weights if weights[x].has_key(word)])

	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist

	return tf_eccd
