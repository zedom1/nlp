import os
from numpy import *
import tensorflow as tf

### hyper-perameters:
tf.flags.DEFINE_string("mode","train","mode")
tf.flags.DEFINE_string("save_path","./model_save","save_path")
tf.flags.DEFINE_string("voca_path",None,"voca_path")
tf.flags.DEFINE_string("embedding_path",None,"embedding_path")

FLAGS = tf.flags.FLAGS

class Config(object):
	batch_size = 2
	learning_rate = 0.01
	keep_prob = 0.8

	rnn_dim = 64
	num_layers = 1
	voca_size = 18
	hidden_size = 100
	embedding_size = 100

	max_length_q = 5
	max_length_a = 5

	max_epoch = 20
	mode = FLAGS.mode
	save_path = FLAGS.save_path
	voca_path = FLAGS.voca_path
	embedding_path = FLAGS.embedding_path

def getConfig():
	return Config()

### input

input_q = [[1,2,3,0,0],[9,8,7,6,0]]
input_a = [[4,5,6,7,0],[1,2,3,0,0]]
labels = [1,0]
seq_length_q = [3,4]
seq_length_a = [4,3]

config = getConfig()
eval_config = getConfig()
eval_config.batch_size = 1
eval_config.mode = "test"

def produce_input(config, input_q, input_a, seq_length_q, seq_length_a, labels):
	input_q = reshape(input_q, [-1])
	input_a = reshape(input_a, [-1])

	input_q = tf.convert_to_tensor(input_q, dtype=tf.int32)
	input_a = tf.convert_to_tensor(input_a,  dtype=tf.int32)
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
	### embedding
	if (FLAGS.embedding_path is not None) and (FLAGS.voca_path is not None):
		initializer = loadEmbedding()
	else:
		initializer = tf.random_uniform_initializer(-0.25, 0.25)

	embedding = tf.get_variable(name = "embedding",shape = [config.voca_size, config.embedding_size], initializer = initializer )

	embedding_q = tf.nn.embedding_lookup(embedding, input_q)
	embedding_a = tf.nn.embedding_lookup(embedding, input_a)

	return embedding_q,embedding_a

### model
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

with tf.Graph().as_default():
	input_q, input_a, seq_length_q, seq_length_a, labels = produce_input(config, input_q, input_a, seq_length_q, seq_length_a, labels)
	embedding_q, embedding_a = embedding(input_q, input_a)
	probs, loss = dual_encoder(config, embedding_q, embedding_a, seq_length_q, seq_length_a, labels)
	optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
	sv = tf.train.Supervisor(logdir=config.save_path)
	config_proto = tf.ConfigProto(allow_soft_placement=True)
	saver = sv.saver
	with sv.managed_session(config=config_proto) as session:
		for i in range(config.max_epoch):
			pprobs,ploss,_ = session.run([probs,loss, optimizer])
			print(ploss)

		if config.save_path is not None and os.path.exists(config.save_path):
			print("Saving model to %s." % config.save_path)
			saver.save(session, os.path.join(config.save_path,"model.ckpt"), global_step=sv.global_step)
