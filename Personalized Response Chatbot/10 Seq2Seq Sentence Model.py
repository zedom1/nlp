#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from preppy import Preppy
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


# Make a dataset by reading the train 

# In[2]:


def expand(x):
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']),0)
    return x
def deflate(x):
    x['length'] = tf.squeeze(x['length'])
    return x


# In[3]:


tf.reset_default_graph()
dataset_train = tf.data.TFRecordDataset(['./data/seq2seq/train.tfrecord']).map(Preppy.parse)
dataset_val = tf.data.TFRecordDataset(['./data/seq2seq/val.tfrecord']).map(Preppy.parse)


# In[4]:


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))


# In[5]:


batched_train = dataset_train.map(expand).padded_batch(8,padded_shapes={
    "length":1,
    "sentence":tf.TensorShape([None])
}).map(deflate)

batched_val = dataset_val.map(expand).padded_batch(8,padded_shapes={
    "length":1,
    "sentence":tf.TensorShape([None])
}).map(deflate)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, batched_train.output_types, batched_train.output_shapes)

next_item = iterator.get_next()


# In[6]:


iterator_train = batched_train.make_initializable_iterator()
iterator_val = batched_val.make_initializable_iterator()

handle_train = sess.run(iterator_train.string_handle())
handle_val = sess.run(iterator_val.string_handle())


# In[7]:



class Model():
    def __init__(self, inputs, rnn_size, vocab_size, embedding_dim, grad_clip):
        
        sentence =  inputs['sentence']
        length = inputs['length']
        
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)

        # define embedding layer
        with tf.variable_scope('embedding'):
            self.embedding = tf.Variable(
            	tf.truncated_normal(shape=[vocab_size, embedding_dim], stddev=0.1), 
                name='embedding')

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(rnn_size)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(self.embedding, sentence)

        encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, length, dtype=tf.float32)
        
        self.target_input_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
        with tf.device('/cpu:0'):
            target_embeddeds = tf.nn.embedding_lookup(self.embedding, self.target_input_ids)
        helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(vocab_size)
            decoder_cell = self._get_simple_lstm(rnn_size)
            decoder = BasicDecoder(decoder_cell, helper, self.encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        targets = tf.reshape(sentence, [-1])
        logits_flat = tf.reshape(logits.rnn_output, [-1, vocab_size])
		
        self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def _get_simple_lstm(self, rnn_size):
        lstm_layers = tf.contrib.rnn.LSTMCell(rnn_size) 
        # lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
        return lstm_layers


# In[8]:


M = Model(next_item, rnn_size = 2, vocab_size = 218, embedding_dim = 2, grad_clip=3)
sess.run(tf.global_variables_initializer())


# In[ ]:


for epoch in range(3):
    sess.run(iterator_train.initializer)
    print("Training %d/3"%(epoch))
    while True:
        try:
            _,loss = sess.run([M.train_op, M.cost],feed_dict={handle: handle_train, M.lr:0.0001})
            print(loss)
        except:
            pass
    print("Validation %d/3"%(epoch))
    sess.run(iterator_val.initializer)
    while True:
        try:
            loss = sess.run(M.cost, feed_dict={handle: handle_val})
            print(loss)
        except:
            pass


# In[ ]:




