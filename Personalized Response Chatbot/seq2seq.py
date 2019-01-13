#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense

class Seq2SeqModel(object):

    def __init__(self, rnn_size, vocab_size, embedding_dim, grad_clip):
        # define inputs
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        # define embedding layer
        with tf.variable_scope('embedding'):
            self.embedding = tf.Variable(
            	tf.truncated_normal(shape=[self.vocab_size, self.embedding_dim], stddev=0.1), 
                name='embedding')

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(self.rnn_size)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)

        encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, self.seq_length, dtype=tf.float32)

        self.target_input_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
        with tf.device('/cpu:0'):
            target_embeddeds = tf.nn.embedding_lookup(self.embedding, self.target_input_ids)
        helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(self.vocab_size)
            decoder_cell = self._get_simple_lstm(self.rnn_size)
            decoder = BasicDecoder(decoder_cell, helper, self.encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        targets = tf.reshape(self.input_x, [-1])
        logits_flat = tf.reshape(logits.rnn_output, [-1, self.vocab_size])

        self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def _get_simple_lstm(self, rnn_size):
        lstm_layers = tf.contrib.rnn.LSTMCell(rnn_size) 
        # lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
        return lstm_layers

model = Seq2SeqModel(10, 20, 30)
train_op = model.train(3)

