from numpy import * 
from my import reader

counter = 0

def genPredict(result, save_file):
	result = reshape(result,[-1,47,9175])
	f = open("./test_char.txt").read().strip().split("\n")
	word_to_id = reader.get_dict()

	proba = []
	global counter
	for lineind in range(shape(result)[0]):
		line = f[counter+lineind].split()
		counter+=1
		length = len(line)
		i = 1
		temproba = []
		while i<length:
			ind = 0
			if line[i] in word_to_id:
				ind = word_to_id[line[i]]
			temproba.append([line[i],log(result[lineind][47-length + i][ind])])
			i += 1
		proba.append(temproba)

	for i in proba:
		save_file.write(str(i)+"\n")
		print(i)
	save_file.write("===================\n")

def main():
	f = open("./result_proba.txt").read().split("\n")
	a = []
	count = 0
	for line in f:
		a = a+line.strip().split(" ")
		count += 1
		if count==47:
			count = 0
			a = array(a,dtype=float32)
			genPredict(a)
			a = []