import os
import time
from random import randint
import collections
from numpy import *
import tensorflow as tf

### hyper-perameters:
tf.flags.DEFINE_string("mode","test","mode")
tf.flags.DEFINE_string("train_path","./corpus/conv_filter.txt","train_path")
tf.flags.DEFINE_string("test_path","./test.txt","test_path")
tf.flags.DEFINE_string("save_path","./model_save","save_path")
tf.flags.DEFINE_string("voca_path","./voca.txt","voca_path")
tf.flags.DEFINE_string("embedding_path",None,"embedding_path")

FLAGS = tf.flags.FLAGS

word_to_id = {}
id_to_word = {}
back_up_a = []
back_up_length = []
probList = []

class Config(object):
	batch_size = 16
	learning_rate = 0.001
	keep_prob = 0.8

	rnn_dim = 256
	num_layers = 2
	voca_size = 3537
	hidden_size = 100
	embedding_size = 200

	max_length_q = 20
	max_length_a = 20

	max_epoch = 10
	mode = FLAGS.mode

def getConfig():
	return Config()

config = getConfig()
eval_config = getConfig()
eval_config.batch_size = 1
eval_config.keep_prob = 1
eval_config.mode = "test"

def build_vocab(filename):
	voca_path = FLAGS.voca_path 
	global word_to_id

	if voca_path is not None and os.path.exists(voca_path):
		word_to_id = eval(open(voca_path).read())
		print("Vocabulary size: %d"%(len(word_to_id)))
		return word_to_id
	
	data = open(filename).read().replace("\n","")

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	count_pairs = [(a,b) for (a,b) in count_pairs if int(b)>=10]

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	word_to_id["UNK"] = len(word_to_id)
	word_to_id["None"] = len(word_to_id)

	if voca_path is not None:
		f = open(voca_path,"w")
		f.write(str(word_to_id))
		f.close()
	print("Vocabulary size: %d"%(len(word_to_id)))

	return word_to_id

def file_to_id(filename):
	word_to_id = build_vocab(filename)
	data = open(filename).read().strip().split("\n")
	unk = word_to_id["UNK"]
	padding = word_to_id["None"]
	inputs = []
	seq_length = []
	for line in data:
		ll = []
		for word in line:
			if word in word_to_id:
				ll.append(word_to_id[word])
			else:
				ll.append(unk)
		
		seq_length.append(len(ll))
		ll += [padding]*(20-len(ll))
		inputs.append(ll)
	return array(inputs), array(seq_length)

def id_to_sentence(sentence):
	global id_to_word, word_to_id
	if len(id_to_word)==0:
		id_to_word = {v:k for k,v in word_to_id.items()}
	result = [id_to_word[i] for i in sentence ]
	result = "".join(result)
	return result

def get_back_up():
	global back_up_a, back_up_length
	inputs, seq_length = file_to_id(FLAGS.train_path)
	back_up_a = input_a = inputs[range(1,len(inputs),2)]
	back_up_length = seq_length_a = seq_length[range(1,len(seq_length),2)]
	return input_a, seq_length_a

def get_input(mode):


	if mode == "train":
		inputs, seq_length = file_to_id(FLAGS.train_path)
		input_q = inputs[range(0,len(inputs),2)]
		input_a = inputs[range(1,len(inputs),2)]
		seq_length_q = seq_length[range(0,len(seq_length),2)]
		seq_length_a = seq_length[range(1,len(seq_length),2)]
		
		num_samples = len(seq_length)//2

		labels = ones([num_samples],dtype=int32)

		randind = random.randint(num_samples, size=num_samples*2)
		for i in range(num_samples*2):
			ind = int(randind[i])
			input_q = concatenate([input_q, reshape(input_q[i//2],[1,-1])], axis=0)
			seq_length_q = concatenate([seq_length_q, reshape(seq_length_q[i//2],[-1])], axis=0)
			while ind == i//2:
				ind = random.randint(num_samples, size=1)[0]
			input_a = concatenate([input_a, reshape(input_a[ind],[1,-1])], axis=0)
			seq_length_a = concatenate([seq_length_a, reshape(seq_length_a[ind],[-1])], axis=0)
			labels = concatenate([labels,array([0])])

		reind = random.permutation(len(labels))

		return input_q[reind], input_a[reind], seq_length_q[reind], seq_length_a[reind], labels[reind]

	elif mode == "test":
		input_a, seq_length_a = get_back_up()
		input_q, seq_length_q = file_to_id(FLAGS.test_path)

		len_a = shape(input_a)[0]
		len_q = shape(input_q)[0]

		input_q = repeat(input_q, len_a, axis= 0)
		seq_length_q = repeat(seq_length_q, len_a, axis= 0)

		input_a = tile(input_a, (len_q,1))
		seq_length_a = tile(seq_length_a, (len_q,1))
		# labels no use
		labels = array(seq_length_q)
		return input_q, input_a, seq_length_q, seq_length_a, labels

def produce_input(config, input_q, input_a, seq_length_q, seq_length_a, labels):
	input_q = tf.reshape(input_q, [-1])
	input_a = tf.reshape(input_a, [-1])

	input_q = tf.cast(input_q, dtype=tf.int32)
	input_a = tf.cast(input_a,  dtype=tf.int32)
	seq_length_q = tf.convert_to_tensor(seq_length_q,  dtype=tf.int32)
	seq_length_a = tf.convert_to_tensor(seq_length_a,  dtype=tf.int32)
	labels = tf.convert_to_tensor(labels,  dtype=tf.int32)

	batch_size = config.batch_size

	data_len_q = tf.size(input_q)
	batch_len_q = data_len_q // batch_size
	data_len_a = tf.size(input_a)
	batch_len_a = data_len_a // batch_size

	input_q = tf.reshape(input_q[0 : batch_size * batch_len_q],[batch_size, batch_len_q])
	seq_length_q = tf.reshape(seq_length_q[0 : batch_size * (batch_len_q//config.max_length_q)],[batch_size, (batch_len_q//config.max_length_q)])
	
	input_a = tf.reshape(input_a[0 : batch_size * batch_len_a],[batch_size, batch_len_a])
	seq_length_a = tf.reshape(seq_length_a[0 : batch_size * (batch_len_a//config.max_length_a)],[batch_size, (batch_len_a//config.max_length_a)])
	
	labels = tf.reshape(labels[0: batch_size * (batch_len_q//config.max_length_q)], [batch_size,(batch_len_q//config.max_length_q)])
	
	epoch_size = (batch_len_q) // config.max_length_q
	epoch_size = tf.identity(epoch_size, name="epoch_size")

	i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
	
	input_q = tf.strided_slice(input_q, [0, i * config.max_length_q], [batch_size, (i + 1) * config.max_length_q])
	input_q.set_shape([batch_size, config.max_length_q])

	seq_length_q = tf.strided_slice(seq_length_q, [0, i],[batch_size, i+1])
	seq_length_q = tf.reshape(seq_length_q, [-1])

	input_a = tf.strided_slice(input_a, [0, i * config.max_length_a], [batch_size, (i + 1) * config.max_length_a])
	input_a.set_shape([batch_size, config.max_length_a])

	seq_length_a = tf.strided_slice(seq_length_a, [0, i],[batch_size, i+1])
	seq_length_a = tf.reshape(seq_length_a, [-1])

	labels = tf.strided_slice(labels, [0, i], [batch_size, (i + 1)])
	labels.set_shape([batch_size, 1])

	return input_q, input_a, seq_length_q, seq_length_a, labels

def embedding(input_q, input_a):
	if (FLAGS.embedding_path is not None) and (FLAGS.voca_path is not None):
		initializer = loadEmbedding()
	else:
		initializer = tf.random_uniform_initializer(-0.25, 0.25)

	embedding = tf.get_variable(name = "embedding",shape = [config.voca_size, config.embedding_size], initializer = initializer )

	embedding_q = tf.nn.embedding_lookup(embedding, input_q)
	embedding_a = tf.nn.embedding_lookup(embedding, input_a)

	return embedding_q,embedding_a

def dual_encoder(config, embedding_q,embedding_a, seq_length_q, seq_length_a, labels):
	with tf.name_scope("rnn"):
		def make_cell():
			cell = tf.contrib.rnn.LSTMBlockCell(config.rnn_dim, forget_bias=1.0)
			if config.mode=="train" and config.keep_prob<1:
				cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
			return cell

		cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
		#cell_bw = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

		outputs, state = tf.nn.dynamic_rnn(cell, inputs = tf.concat([embedding_q,embedding_a],0), sequence_length = tf.concat([seq_length_q, seq_length_a], 0), dtype=tf.float32)
		encoding_q, encoding_a = tf.split(state[0].h, 2, axis=0)

	with tf.name_scope("prediction"):
		M = tf.get_variable(name="M", shape=[config.rnn_dim,config.rnn_dim], initializer = tf.truncated_normal_initializer())

		score = tf.matmul(encoding_q, M)
		score = tf.expand_dims(score, 2)
		encoding_a = tf.expand_dims(encoding_a, 2)
		score = tf.matmul(score, encoding_a,True)
		logits = tf.squeeze(score,[-1])
		probs = tf.sigmoid(logits)

		if config.mode == "test":
			return probs, None

		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.reshape(tf.to_float(labels),[config.batch_size,-1]))

	mean_loss = tf.reduce_mean(loss)

	return probs, mean_loss

class Model(object):
	probs = None
	loss = None
	optimizer = None
	epoch_size = 0
	config = None

	def __init__(self, probs, loss, optimizer, epoch_size, config):
		self.probs = probs
		self.loss = loss
		self.optimizer = optimizer
		self.epoch_size = epoch_size
		self.config = config

def build_model(config):

	print("Getting Inputs....")
	input_q, input_a, seq_length_q, seq_length_a, labels = get_input(config.mode)
	print("Getting Inputs Finish")

	print("Producing Batches....")
	epoch_size = shape(labels)[0]//config.batch_size
	input_q, input_a, seq_length_q, seq_length_a, labels = produce_input(config,input_q, input_a, seq_length_q, seq_length_a, labels)
	embedding_q, embedding_a = embedding(input_q, input_a)
	print("Producing Batches Finish")
	
	probs, loss = dual_encoder(config, embedding_q, embedding_a, seq_length_q, seq_length_a, labels)
	if config.mode == "train":
		optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
	else:
		optimizer = None
	return Model(probs,loss,optimizer,epoch_size, config)

def run_epoch(model, session):
	start_time = time.time()
	costs = 0.0
	iters = 0
	total_loss = 0.0
	
	for i in range(model.epoch_size):
		if model.config.mode == "train":
			pprobs,ploss,_ = session.run([model.probs, model.loss, model.optimizer])
			total_loss += ploss
			iters += model.config.max_length_q

			if i % (model.epoch_size // 10) == 30:
				print("%.3f cost1: %.3f cost2 : %.3f speed: %.0f wps" %
				(
					i * 1.0 / model.epoch_size, 
					exp(total_loss / iters),
					ploss,
					iters * model.config.batch_size / (time.time() - start_time))
				)
		else:
			global probList
			pprobs = session.run(model.probs)
			probList.append((i,pprobs))

	return total_loss/model.epoch_size

def handle_test():
	global probList, back_up_a, back_up_length
	f = open(FLAGS.test_path).read().strip().split("\n")
	length = shape(back_up_a)[0]
	for i in range(len(f)):
		question = f[i]
		print("="*10+"\n"+question)
		ans = probList[i*length:(i+1)*length]
		ans = sorted(ans, key=lambda x: (-x[1][0][0], x[0]))[:3]
		for sentence in ans:
			ind = (sentence[0])%length
			prob = sentence[1][0][0]
			sentence = back_up_a[ind][:back_up_length[ind]]
			sentence = id_to_sentence(sentence)
			print("ans: %s   prob: %.5f"%(sentence,prob))

with tf.Graph().as_default():
	
	if FLAGS.mode == "train":
		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None) as scope:
				trainModel = build_model(config)
				scope.reuse_variables()
		
		with tf.name_scope("Test"):
			with tf.variable_scope("Model", reuse=True) as scope:
				testModel = build_model(eval_config)
		
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		config_proto = tf.ConfigProto(allow_soft_placement=True)
		saver = sv.saver
		
		with sv.managed_session(config=config_proto) as session:
			
			for i in range(config.max_epoch):
				loss = run_epoch(trainModel, session)
				print("Epoch : %d  Loss: %.3f "%(i,loss))

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