import numpy as np
import pandas as pd
import collections
import re
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences

def rep(sentence, mode = 0):

	if mode == 0:
		sentence = sentence.replace("__eot__", "").replace("__eou__", "\n")
	elif mode == 1:
		sentence = sentence.replace("__eot__", "").replace("__eou__", "")
	sentence = re.sub(r'\W', ' ', sentence)
	return sentence.strip().lower()

def processUbuntuTrain(filepath):
	data = pd.read_csv(filepath).values

	context = data[:,0]
	context = np.array([rep(i, mode = 0) for i in context]).reshape(-1)
	utterance = data[:,1]
	utterance = np.array([rep(i, mode = 1) for i in utterance]).reshape(-1)
	label = data[:,2].reshape(-1).astype(int)
	
	return context, utterance, label

def produceText(path, writePath):
	data = pd.read_csv(path).values

	context = np.concatenate( (data[:,0].reshape(-1), data[:,1].reshape(-1)) )
	context = np.array([rep(i, mode = 0) for i in context]).reshape(-1)

	sentence = context.tolist()
	context = ' '.join(sentence).split("\n")
	sentence = ' '.join(sentence).replace("\n","").split()
	counter = collections.Counter(sentence)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	count_pairs = [(a,b) for (a,b) in count_pairs if int(b)>=5]
	count_pairs = [a for (a,b) in count_pairs]
	answers = []
	with open(writePath, mode='wt', encoding='utf-8') as myfile:
		print(len(context))
		for i in range(len(context)):
			if i%10000 == 0:
				print(i)
			line = context[i]
			c = [ ["UNK", x][x in count_pairs] for x in line.split()]
			myfile.write(' '.join(c))
			myfile.write("\n")


def processUbuntuDev(filepath):
	data = pd.read_csv(filepath).values.tolist()
	
	for line in data:
		line[0] = rep(line[0])
		for i in range(1,len(line)):
			line[i] = rep(line[i], mode=1)
			
	label = []
	numSamples = len(data)
	numUtter = len(data[0])-1
	for i in range(numUtter):
		if i == 0:
			label += [1]*numSamples
		else:
			label += [0]*numSamples
	label = np.array(label).reshape(-1)
	context = np.tile(np.array(data)[:,0].reshape(-1),numUtter)
	utterance = np.array(data)[:,1].reshape(-1)
	for i in range(2,numUtter+1):
		utterance = np.concatenate((utterance, np.array(data)[:,1].reshape(-1)))

	return context, utterance, label


def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length

def multi_sequences_padding(all_sequences, config):
    max_num_utterance = config.max_num_utterance
    max_sentence_len = config.max_length_q
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, maxlen=max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length

def Evaluation(score, count = 5):
	pairs = [[i, score[i]] for i in range(len(score))]
	sorted_pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
	ans = [i[0] for i in sorted_pairs[:count]]
	return int(0 in ans)
