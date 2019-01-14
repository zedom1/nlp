import codecs
from random import randint

f = codecs.open("./data/rank/rank.txt","r","utf-8").read().strip().split("\n")
f1 = codecs.open("./data/rank/val.txt","w","utf-8")

sentences = []
i = 0
while i < len(f):
	temline = f[i].strip().split("\t")
	user = int(temline[0])
	query = temline[2]
	response = temline[-1]
	for _ in range(1,10):
		temsen = f[i+_].strip().split("\t")
		if int(temsen[1]) == 1:
			response = temsen[-1] + "\t" + response
		else:
			response += "\t" + temsen[-1]
	i += 10
	sentences.append("{user}\t{query}\t{response}".format(user=user, query=query, response=response))

f1.write("\n".join(sentences))
f1.close()