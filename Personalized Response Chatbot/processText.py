import codecs
from random import randint

f = codecs.open("./data/seq2seq/sentences.txt","r","utf-8").read().strip().split("\n")
f1 = codecs.open("./data/user/user.txt","w","utf-8")

sentences = []

for line in f:
	temline = line.strip()
	user = randint(0,100)
	label = randint(0,1)
	sentences.append("{user}\t{label}\t{sentence}".format(user=user, label=label, sentence=temline))

f1.write("\n".join(sentences))
f1.close()