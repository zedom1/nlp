import pickle
import numpy as np
import tensorflow as tf
from preppy import Preppy
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import dense


class RankModel():
    def __init__(self, params):
        
        concat_shape = params["user_embedding"] + params["sentence_embedding"]
        self.query = tf.placeholder(shape=[None, params["sentence_embedding"]],dtype=tf.float32)
        self.response = tf.placeholder(shape=[None, params["sentence_embedding"]],dtype=tf.float32)
        self.user = tf.placeholder(shape=[None],dtype=tf.int32)
        self.label = tf.placeholder(shape=[None],dtype=tf.int32)
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        
        with open(params["user_embedding_file"],"rb") as emb_f:
	        user_emb = pickle.load(emb_f)
        # define embedding layer
        self.user_embedding = tf.Variable(
			initial_value = user_emb,
			trainable = False)
        self.embedded_user = tf.nn.embedding_lookup(self.user_embedding, self.user)

        q_u_inputs = tf.concat([self.query, self.embedded_user], 1)
        q_u_inputs.set_shape([None, concat_shape])

        r_u_inputs = tf.concat([self.response, self.embedded_user], 1)
        r_u_inputs.set_shape([None, concat_shape])

        p_cond_u = dense(inputs=q_u_inputs, units=params["p_size"], activation=tf.tanh)
        r_cond_u = dense(inputs=r_u_inputs, units=params["r_size"], activation=tf.tanh)

        q_r_cond = tf.concat([p_cond_u, r_cond_u], 1)

        f_p_r_cond_u = dense(inputs=q_r_cond, units=params["f_size"], activation=tf.tanh)
        s_u_r = dense(inputs=r_cond_u, units=params["s_size"], activation=tf.tanh)
        
        f_s = tf.concat([f_p_r_cond_u, s_u_r], 1)

        logits = dense(inputs=f_s, units=1)

        self.predict = tf.sigmoid(logits)
        
        float_label = tf.to_float(self.label)
        logits = tf.squeeze(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=float_label, logits=logits))
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        """
        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), params["grad_clip"])

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        """
        