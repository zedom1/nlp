import codecs

f = codecs.open("./data/rank/rank.txt","r","utf-8").read().strip().split("\n")
f1 = codecs.open("./data/seq2seq/text.txt","w","utf-8")

sentences = set()

for line in f:
	temline = line.split("\t")[2:]
	sentences.add(temline[0])
	sentences.add(temline[1])

f1.write("\n".join(sentences))
f1.close()