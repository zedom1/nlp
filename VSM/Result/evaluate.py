# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os import walk
import matplotlib.pyplot as plt 
"""
microaverage = accuracy = correct / total

macroaverage = avg(F1)

F1(C) = 2*precision*recall / (precision + recall)
precision = TP / (TP + FN)	： TP/预测为C的数目
recall = TP / (TP + TN)		： TP/真实为C的数目

TP: true positive  属于C被分到C（正确）
TN: true nagative  属于C被分到其它（错误）

FP: false positive 不属于C被正确分类（正确）
FN: false nagative 不属于C被分到C（错误）

/media/zedom/Study/NLP/Student Research Project/Part1 : Learning Project/Data/Reuters_train.txt
/media/zedom/Study/NLP/Student Research Project/Part1 : Learning Project/Data/Reuters_test.txt

"""
def evaluate(pathname, filename):
	global filedict
	filedict[filename] = {}
	TP = {}
	TN = {}
	FN = {}
	file = open(pathname+"/"+filename).read().strip().split("\n")
	total = len(file)
	for i in file:
		line = i.split(" ")
		#print line
		predict = int(float(line[0]))
		label = int(float(line[1]))
		if not TP.has_key(label):
			TP[label]=0
		if not TN.has_key(label):
			TN[label]=0
		if not FN.has_key(predict):
			FN[predict]=0
		if not FN.has_key(label):
			FN[label]=0
		if predict == label:
			TP[label] += 1
		else:
			TN[label] += 1
			FN[predict] += 1

	F1 = []
	for i in TP:
		Pre = TP[i]*1.0 / (TP[i] + FN[i])
		Re = TP[i]*1.0 / (TP[i] + TN[i])
		F1.append(2.0*Pre*Re / (Pre + Re))
	print filename + "\ttrain: 5485  / test: "+str(total)
	MicroF1 = sum(TP.values())/float(total)
	MacroF1 = sum(F1)/float(len(F1))
	print "MicroF1 = "+str(MicroF1) 
	print "MacroF1 = "+str( MacroF1 ) 
	filedict[filename]["MicroF1"] = MicroF1
	filedict[filename]["MacroF1"] = MacroF1

filedict = {}
for (dirpath,dirnames,filenames) in walk("./"):
	for file in filenames:
		if not file.endswith(".txt"):
			continue
		print file
		evaluate(dirpath, file)
		
plt.plot(filedict.keys(), [filedict[x]["MicroF1"] for x in filedict.keys()],"x-",label = "MicroF1")
plt.plot(filedict.keys(), [filedict[x]["MacroF1"] for x in filedict.keys()],"+-",label = "MacroF1")
plt.legend(loc='upper right')
plt.show()