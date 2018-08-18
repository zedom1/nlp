
#f = open("result_proba.txt").read().strip().replace("], ['","\n").replace("[['","").replace("',","").replace("]]","").split("\n")

def generate(sentence, index, test_path):
	f = sentence.strip().split("\n")
	f1 = open(test_path+"_ans").read().strip().split("\n")
	fresult = open("./report/report_"+str((test_path.split("/")[-1]).split("_")[-1])+".txt","a")
	ss = "=======\nIndex:%d\n"%index
	i = 0
	i1 = 0
	parti = 0
	flag = 0
	ls = []
	predictInd = []

	tp1 = 0 
	fp1 = 0
	tp05 = 0
	fp05 = 0

	tn = 0
	fn = 0

	while i<len(f):
		if f[i] == "===================":
			i += 1
			parti = 0
			ans = list(map(int,f1[i1].split()))
			if ans[0] != -1:
				ans = [a-1 for a in ans]
			i1 += 1
			#print("Ans: %s"%ans)
			#print("PreIndex:%s"%predictInd)
			if flag == 0:
				print("-1")
				if ans[0] == -1:
					tn += 1
				else:
					fn += 1
			if len(predictInd)>=1:
				if len(predictInd) > len(ans):
					fp1 += 1
				elif len(predictInd) == len(ans):
					if sorted(predictInd) == sorted(ans):
						tp1 += 1
					else:
						flag = 0
						for temind in predictInd:
							if temind in ans:
								flag = 1
								break
						if flag == 1:
							tp05 += 1
							fp05 += 1
						else:
							fp1 += 1
				else:
					flag = 1
					for temind in predictInd:
						if temind not in ans:
							flag = 0
					if flag == 1:
						tp05 += 1
					else:
						fp1 += 1
				#for ll in ls:
					#print(ll)
			ls = []
			#print("=================")
			flag = 0
			predictInd = []
			continue
		line = f[i].split()
		if float(line[1])<-10.3:
			ls.append([line])
			predictInd.append(parti)
			flag = 1
		i += 1
		parti += 1

	ss+=("========= Report ===========\n")
	ss+=("tp1:%d\n"%tp1)
	ss+=("tp05:%d\n"%tp05)
	ss+=("fp1:%d\n"%fp1)
	ss+=("fp05:%d\n"%fp05)
	ss+=("tn:%d\n"%tn)
	ss+=("fn:%d\n"%fn)

	tp = tp1 + tp05*0.5
	fp = fp1 + fp05*0.5
	ss+=("=========\n")
	if (tp+fp)>=0.1 and (tp+fn)>=0.1 and tp>0.1:
		pre = (tp/(tp+fp))
		rec = (tp/(tp+fn))
		ss+=("Precision\t=\t%f\n"%pre)
		ss+=("Recall\t\t=\t%f\n"%rec)
		ss+=("Accuracy\t=\t%f\n"%((tp+tn)/(tp+fp+tn+fn)))
		ss+=("F1\t\t=\t%f\n"%(2*rec*pre/(rec+pre)))
		ss+=("========= Report ===========\n")
	fresult.write(ss)
	fresult.close()
