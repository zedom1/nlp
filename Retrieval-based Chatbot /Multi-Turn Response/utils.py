import numpy as np
import pandas as pd
import re
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences

def rep(sentence, mode = 0):
	if mode == 0:
		#train
		sentence = sentence.replace("__eot__", "").replace("__eou__", "\n")
	else:
		sentence = sentence.replace("__eot__", "").replace("__eou__", "")
	sentence = re.sub(r'[\(\)\,><=\+\'\-\?\!\.]', ' ', sentence)
	return sentence.strip()

def processUbuntuTrain(filepath):
	data = pd.read_csv(filepath)

	context = np.array(data["Context"]).tolist()
	context = np.array([rep(i, mode = 0) for i in context]).reshape(-1)
	utterance = np.array(data["Utterance"])
	utterance = np.array([rep(i, mode = 1) for i in utterance]).reshape(-1)
	label = np.array(data["Label"]).reshape(-1)
	
	return context, utterance, label

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

"""
a = processUbuntuTrain("./Data/train0.csv")
print(type(a[2]))
print((a[2]))
print("/*-"*10)
b = processUbuntuDev("./Data/dev0.csv")
print(type(b[2]))
print((b[2]))
"""