import os
import time
from random import randint
import collections
from utils import *
from numpy import *
import tensorflow as tf

### hyper-perameters:
tf.flags.DEFINE_string("mode","train","mode")
tf.flags.DEFINE_string("train_path","./Data/Train/train_0","train_path")
tf.flags.DEFINE_string("dev_path","./Data/dev.csv","dev_path")
tf.flags.DEFINE_string("test_path","./Data/test.csv","test_path")
tf.flags.DEFINE_string("save_path","./model/0","save_path")
tf.flags.DEFINE_string("voca_path","./voca.txt","voca_path")
tf.flags.DEFINE_string("embedding_path",None,"embedding_path")

FLAGS = tf.flags.FLAGS

word_to_id = {}
id_to_word = {}
evalProbs = []
init = None
voca_size = 0

class Config(object):
	batch_size = 200
	learning_rate = 0.001
	keep_prob = 1.0

	rnn_dim = 200
	embedding_size = 200

	max_length_q = 50
	max_length_a = 50
	max_num_utterance = 1

	max_epoch = 20
	mode = FLAGS.mode

def getConfig():
	return Config()

config = getConfig()
eval_config = getConfig()
eval_config.keep_prob = 1
eval_config.mode = "dev"

def build_vocab(dataTuple):
	voca_path = FLAGS.voca_path 
	global word_to_id, voca_size

	if voca_path is not None and os.path.exists(voca_path):
		word_to_id = eval(open(voca_path).read())
		print("Vocabulary size: %d"%(len(word_to_id)))
		voca_size = len(word_to_id)
		return word_to_id
	
	data = []
	for i in dataTuple:
		data += list(i.reshape(-1))
		#print(list(i.reshape(-1)))
	data = ' '.join(data).replace("\n","").split()
	#print(data)
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	count_pairs = [(a,b) for (a,b) in count_pairs if int(b)>=10]

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(1, len(words)+1)))
	word_to_id["UNK"] = len(words)+1
	word_to_id["."] = 0

	if voca_path is not None:
		f = open(voca_path,"w")
		f.write(str(word_to_id))
		f.close()
	print("Vocabulary size: %d"%(len(word_to_id)))
	voca_size = len(word_to_id)
	return word_to_id

def file_to_id(data, mode = 0):
	unk = word_to_id["UNK"]
	inputs = []
	seq_lengths = []
	for sequences in data:
		sequences = sequences.split("\n")
		single_input = []
		seq_length = []
		for line in sequences:
			line = line.split()
			ll = []
			for word in line:
				if word in word_to_id:
					ll.append(word_to_id[word])
				else:
					ll.append(unk)
			
			seq_length.append(len(ll))
			single_input.append(ll)

		inputs.append(single_input)
		seq_lengths.append(seq_length)
	if mode == 1:
		inputs = array(inputs).reshape(-1,1)
		seq_lengths = array(seq_lengths).reshape(-1,1)
	return inputs, seq_lengths

def id_to_sentence(sentence):
	global id_to_word, word_to_id
	if len(id_to_word)==0:
		id_to_word = {v:k for k,v in word_to_id.items()}
	result = [id_to_word[i] for i in sentence ]
	result = "".join(result)
	return result

def get_input(mode, path = None):

	if mode == "train":
		context, utterance, labels = processUbuntuTrain(path)
		build_vocab([context, utterance])
	
	else:
		context, utterance, labels = processUbuntuDev(FLAGS.dev_path)
		build_vocab(None)

	context, _ = file_to_id(context)
	utterance, seq_length_u = file_to_id(utterance, mode = 1)
	contexts, seq_length_c = multi_sequences_padding(context, config)
	utterance = [i[0] for i in utterance]
	utterance = pad_sequences(utterance, padding='post', maxlen=config.max_length_q)

	return contexts, utterance, seq_length_c, seq_length_u, labels

def produce_input(config, input_q, input_a, seq_length_q, seq_length_a, labels):
	input_q = reshape(input_q, [-1])
	input_a = reshape(input_a, [-1])
	seq_length_q = reshape(seq_length_q, [-1])
	seq_length_a = reshape(seq_length_a, [-1])
	labels = reshape(labels, [-1])
	batch_size = config.batch_size

	data_len_q = len(input_q)
	batch_len_q = data_len_q//config.max_length_q//config.max_num_utterance // batch_size * config.max_length_q * config.max_num_utterance
	data_len_a = len(input_a)
	batch_len_a = data_len_a//config.max_length_a // batch_size * config.max_length_a
	input_q = tf.reshape(input_q[0 : batch_size * batch_len_q],[batch_size, batch_len_q])

	seq_length_q = tf.reshape(seq_length_q[0 : batch_size * (batch_len_q//config.max_length_q)], [batch_size, (batch_len_q//config.max_length_q)])
	input_a = tf.reshape(input_a[0 : batch_size * batch_len_a],[batch_size, batch_len_a])

	seq_length_a = tf.reshape(seq_length_a[0 : batch_size * (batch_len_a//config.max_length_a)],[batch_size, (batch_len_a//config.max_length_a)])
	
	labels = tf.reshape(labels[0: batch_size * (batch_len_a//config.max_length_a)], [batch_size,(batch_len_a//config.max_length_a)])
	
	epoch_size = (batch_len_a) // config.max_length_a
	epoch_size = tf.identity(epoch_size, name="epoch_size")

	i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
	
	input_q = tf.strided_slice(input_q, [0, i * config.max_length_q*config.max_num_utterance], [batch_size, (i + 1) * config.max_length_q*config.max_num_utterance])
	input_q = tf.reshape(input_q, [batch_size, config.max_num_utterance, config.max_length_q])

	seq_length_q = tf.strided_slice(seq_length_q, [0, i*config.max_num_utterance],[batch_size, (i+1)*config.max_num_utterance])
	seq_length_q = tf.reshape(seq_length_q, [batch_size, config.max_num_utterance])

	input_a = tf.strided_slice(input_a, [0, i * config.max_length_a], [batch_size, (i + 1) * config.max_length_a])
	input_a = tf.reshape(input_a, [batch_size, config.max_length_a])

	seq_length_a = tf.strided_slice(seq_length_a, [0, i],[batch_size, i+1])
	seq_length_a = tf.reshape(seq_length_a, [-1])

	labels = tf.strided_slice(labels, [0, i], [batch_size, (i + 1)])
	labels = tf.reshape(labels, [batch_size])

	return input_q, input_a, seq_length_q, seq_length_a, labels

def embedding(input_u, input_r):
	global voca_size, init
	if (FLAGS.embedding_path is not None) and (FLAGS.voca_path is not None):
		initializer = loadEmbedding()
	else:
		initializer = tf.random_uniform_initializer(-0.25, 0.25)

	embedding = tf.get_variable(name = "embedding_m", shape=(voca_size, config.embedding_size), initializer = initializer )

	embedding_u = tf.nn.embedding_lookup(embedding, input_u)
	embedding_r = tf.nn.embedding_lookup(embedding, input_r)

	return embedding_u, embedding_r

def multiTurnResponse(config, embedding_u, embedding_r, seq_length_u, seq_length_r, labels):

	def make_cell(num = 0):
		if num==0:
			cell = tf.nn.rnn_cell.GRUCell(config.rnn_dim, kernel_initializer=tf.orthogonal_initializer())
		else:
			cell = tf.nn.rnn_cell.GRUCell(num, kernel_initializer=tf.orthogonal_initializer())			
		if config.mode=="train" and config.keep_prob<1:
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
		return cell

	sentence_GRU = make_cell()
	final_GRU = make_cell(50)
	embedding_us = tf.unstack(embedding_u, num=config.max_num_utterance, axis=1)
	seq_length_us = tf.unstack(seq_length_u, num=config.max_num_utterance, axis=1)
	A_matrix = tf.get_variable('A_matrix_v', shape=(config.rnn_dim, config.rnn_dim), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

	embedding_r = tf.cast(embedding_r, tf.float32)
	gru_response, _ = tf.nn.dynamic_rnn(sentence_GRU, embedding_r, sequence_length=seq_length_r, dtype=tf.float32, scope='sentence_GRU')
	embedding_r = tf.transpose(embedding_r, perm=[0, 2, 1])
	gru_response = tf.transpose(gru_response, perm=[0, 2, 1])
	matching_vectors = []
	reuse = None
	for embedding_u, seq_length_u in zip(embedding_us, seq_length_us):
		embedding_u = tf.cast(embedding_u, tf.float32)
		matrix1 = tf.matmul(embedding_u, embedding_r)
		gru_utterance, _ = tf.nn.dynamic_rnn(sentence_GRU, embedding_u, sequence_length=seq_length_u, dtype=tf.float32, scope='sentence_GRU')
		matrix2 = tf.einsum('aij,jk->aik', gru_utterance, A_matrix)
		matrix2 = tf.matmul(matrix2, gru_response)
		matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')

		conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
			kernel_initializer=tf.contrib.keras.initializers.he_normal(),
			activation=tf.nn.relu, reuse=reuse, name='conv')
		pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
			padding='VALID', name='max_pooling') 
		matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			activation=tf.tanh, reuse=reuse, name='matching_v')
		matching_vectors.append(matching_vector)

		if not reuse:
			reuse = True

	_, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), 
		dtype=tf.float32, time_major=True, scope='final_GRU')  # TODO: check time_major
	logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
	y_pred = tf.nn.softmax(logits)
	score = tf.reduce_max(y_pred, axis = 1)
	label_pred = tf.cast(tf.argmax(y_pred, 1),tf.int32)
	labels = tf.cast(labels, tf.int32)
	acc = tf.reduce_mean(tf.cast(tf.equal(label_pred, labels), tf.float32))

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

	return score, acc, loss


class Model(object):
	probs = None
	loss = None
	acc = None
	optimizer = None
	epoch_size = 0
	config = None
	input_q = None 
	input_a = None
	seq_length_q = None
	seq_length_a = None
	labels = None

	def __init__(self, config, input_q, input_a, seq_length_q, seq_length_a, labels, epoch_size):
		
		self.resetInput(input_q, input_a, seq_length_q, seq_length_a, labels, epoch_size)
		
		embedding_q, embedding_a = embedding(self.input_q, self.input_a)
		
		probs, acc, loss = multiTurnResponse(config, embedding_q, embedding_a, self.seq_length_q, self.seq_length_a, self.labels)
		if config.mode == "train":
			optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
		else:
			optimizer = None
		self.probs = probs
		self.loss = loss
		self.acc = acc
		self.optimizer = optimizer
		self.epoch_size = epoch_size
		self.config = config

	def resetInput(self, input_q, input_a, seq_length_q, seq_length_a, labels, epoch_size):
		self.input_q = input_q
		self.input_a = input_a
		self.seq_length_q = seq_length_q
		self.seq_length_a = seq_length_a
		self.labels = labels
		self.epoch_size = epoch_size


def input_model(config, path = None):
	print("Getting Inputs....")
	input_q, input_a, seq_length_q, seq_length_a, labels = get_input(config.mode, path)
	print("Getting Inputs Finish")
	print("Producing Batches....")
	epoch_size = shape(labels)[0]//config.batch_size
	input_q, input_a, seq_length_q, seq_length_a, labels = produce_input(config, input_q, input_a, seq_length_q, seq_length_a, labels)
	print("Producing Batches Finish")

	return input_q, input_a, seq_length_q, seq_length_a, labels, epoch_size


def run_epoch(model, session):
	start_time = time.time()
	costs = 0.0
	iters = 0
	acc = 0.0
	total_loss = 0.0
	fetches = {
		"loss": model.loss,
	}
	
	if model.config.mode == "train":
		fetches["optimizer"] = model.optimizer
		fetches["acc"] = model.acc
	else :
		fetches["probs"] = model.probs
	for i in range(model.epoch_size):
		vals = session.run(fetches)
		total_loss += vals["loss"]
		iters += model.config.max_length_q
		
		if model.config.mode == "train":
			acc += vals["acc"]

			if i % (model.epoch_size // 10) == 0:
				print("%.3f cost : %.3f speed: %.1f wps acc: %.3f" %
				(
					i * 1.0 / model.epoch_size, 
					vals["loss"],
					iters * model.config.batch_size / (time.time() - start_time),
					acc / (iters//model.config.max_length_q),
				))

		else:
			global evalProbs
			evalProbs += list(vals["probs"])

	return total_loss/model.epoch_size

def handleTest():
	global evalProbs
	evalProbs = evalProbs[:len(evalProbs)-len(evalProbs)%10]
	evalProbs = reshape(array(evalProbs), [-1,10])
	
	total = shape(evalProbs)[0]
	r10_5 = 0.0
	r10_2 = 0.0
	r10_1 = 0.0
	for i in evalProbs:
		r10_5 += Evaluation(i, 5)
		r10_2 += Evaluation(i, 2)
		r10_1 += Evaluation(i, 1)

	print("Accuarcy: 5/10: %.3f  2/10: %.3f   1/10: %.3f "%(r10_5/total, r10_2/total, r10_1/total))
	evalProbs = []


with tf.Graph().as_default():
	t_input_q, t_input_a, t_seq_length_q, t_seq_length_a, t_labels, t_epoch_size = input_model(config, FLAGS.train_path+"0")
	d_input_q, d_input_a, d_seq_length_q, d_seq_length_a, d_labels, d_epoch_size = input_model(eval_config)

	#	tf.reset_default_graph()	
	if FLAGS.mode == "train":
		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None) as scope:
				trainModel = Model(config, t_input_q, t_input_a, t_seq_length_q, t_seq_length_a, t_labels, t_epoch_size)
				scope.reuse_variables()
		with tf.name_scope("Dev"):
			with tf.variable_scope("Model", reuse=True) as scope:
				devModel = Model(eval_config, d_input_q, d_input_a, d_seq_length_q, d_seq_length_a, d_labels, d_epoch_size)
		
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		config_proto = tf.ConfigProto(allow_soft_placement=True)
		saver = sv.saver
		
		with sv.managed_session(config=config_proto) as session:
			
			for i in range(config.max_epoch):

				for j in range(9):
					print(j)
					tf.reset_default_graph()
					t_input_q, t_input_a, t_seq_length_q, t_seq_length_a, t_labels, t_epoch_size = input_model(config, FLAGS.train_path+str(j))
					trainModel.resetInput(t_input_q, t_input_a, t_seq_length_q, t_seq_length_a, t_labels, t_epoch_size)

					loss = run_epoch(trainModel, session)
					
					run_epoch(devModel, session)
					handleTest()
				print("Epoch : %d  Loss: %.3f "%(i,loss))
				
				run_epoch(devModel, session)
				handleTest()
				
			if FLAGS.save_path is not None and os.path.exists(FLAGS.save_path):
				print("Saving model to %s." % FLAGS.save_path)
				saver.save(session, os.path.join(FLAGS.save_path,"model.ckpt"), global_step=sv.global_step)

	else:
		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None) as scope:
				testModel = build_model(eval_config)
				scope.reuse_variables()

		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		config_proto = tf.ConfigProto(allow_soft_placement=True)
		saver = sv.saver
		with sv.managed_session(config=config_proto) as session:
			
			run_epoch(testModel, session)
			handle_test()