import pickle
import numpy as np
import tensorflow as tf
from preppy import Preppy
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense

class Seq2SeqModel():
    def __init__(self, params):
        
        self.sentence = tf.placeholder(shape=[None, None],dtype=tf.int32)
        self.length = tf.count_nonzero(self.sentence, 1, dtype=tf.int32)
        
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        
        # define embedding layer
        self.embedding = tf.Variable(
            tf.truncated_normal(shape=[params["vocab_size"], params["embedding_size"]], stddev=0.1), name='embedding')

        # define encoder
        encoder = self._get_simple_lstm(params["hidden_size"], params["n_layers"])

        input_x_embedded = tf.nn.embedding_lookup(self.embedding, self.sentence)

        encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, self.length, dtype=tf.float32)
        
        self.target_input_ids = self.sentence
        self.decoder_seq_length = self.length
        
        target_embeddeds = tf.nn.embedding_lookup(self.embedding, self.target_input_ids)
        helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)

        fc_layer = Dense(params["vocab_size"])
        decoder_cell = self._get_simple_lstm(params["hidden_size"], params["n_layers"])
        decoder = BasicDecoder(decoder_cell, helper, self.encoder_state, fc_layer)

        self.logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        masks = tf.to_float(tf.sequence_mask(self.length))
        self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.logits.rnn_output,
                                      targets=self.sentence,
                                      weights=masks) 
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        """
        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), params["grad_clip"])

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        """
        
    def _get_simple_lstm(self, rnn_size, layer):
        if layer == 1:
            lstm_layers = tf.contrib.rnn.LSTMCell(rnn_size)
        else:
            lstm_layers = []
            for i in rnn_size:
                lstm_layers.append(tf.contrib.rnn.LSTMCell(i))
                lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
        return lstm_layers 
