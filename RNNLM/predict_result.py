from numpy import * 
import reader

counter = 0

def genPredict(result):
	result = reshape(result,[-1,43,len(reader.word_to_id)])
	f = open("./test_char.txt").read().strip().split("\n")
	word_to_id = reader.word_to_id

	proba = []
	for lineind in range(len(f)):
		line = f[lineind].split()
		length = len(line)
		i = 1
		temproba = []
		while i<length:
			ind = 0
			if line[i] in word_to_id:
				ind = word_to_id[line[i]]
			temproba.append([line[i],result[lineind][43-length + i][ind]])
			i += 1
		proba.append(temproba)

	for i in proba:
		print(i)
	print(size(proba))