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
def neighbour(pathname, filename):
	global filedict
	file = open(pathname+"/"+filename).read().strip().split("\n")
	filedict[filename] = {}
	filedict[filename]["label"]=filename.replace("knn_","").replace(".txt","")
	filedict[filename]["x"] = []
	filedict[filename]["y"] = []
	for i in file:
		line = i.split("\t")
		#print line
		x = int(float(line[0]))
		y = (float(line[1]))
		print x
		print y
		filedict[filename]["x"].append(x)
		filedict[filename]["y"].append(y)

filedict = {}
for (dirpath,dirnames,filenames) in walk("./"):
	for file in filenames:
		if not file.endswith(".txt"):
			continue
		print file
		neighbour(dirpath, file)

for i in filedict:
	plt.plot(filedict[i]["x"],filedict[i]["y"],"x-",label =filedict[i]["label"] )

plt.legend(loc='lower right')
plt.show()