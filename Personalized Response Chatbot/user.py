import pickle
import numpy as np
import tensorflow as tf
from preppy import Preppy
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import dense


class UserModel():
    def __init__(self, inputs, params):
        
        feature_size = params["sentence_size"]+params["embedding_size"]
        self.sentence = sentence =  tf.placeholder(shape=[None, params["sentence_size"]],dtype=tf.float32)
        label = inputs["label"]
        user = inputs["user"]
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        
        # define embedding layer
        self.embedding = tf.Variable(
            tf.truncated_normal(shape=[params["num_users"], params["embedding_size"]], stddev=0.1), name='embedding')

        self.embedded_user = tf.nn.embedding_lookup(self.embedding, user)

        inputs = tf.concat([sentence, self.embedded_user], 1)
        inputs.set_shape([None, feature_size])

        layer = dense(inputs=inputs, units=params["hidden_size"], activation=tf.tanh)
        logits = dense(inputs=layer, units=1)
        
        float_label = tf.to_float(label)
        logits = tf.squeeze(logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=float_label, logits=logits))
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        """
        # define train op
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), params["grad_clip"])

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        """
        