from numpy import * 

result = reshape(loadtxt("./result_proba.txt",dtype=float32),[176,43,1630])

f = open("./test_char.txt").read().strip().split("\n")
word_to_id = eval(open("./word_id.txt").read())

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