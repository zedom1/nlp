# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import numpy as np
from numpy import *

Py3 = sys.version_info[0] == 3
length = 0
sequence_length = []
word_to_id = {}

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      a = f.read().strip().split("\n")
      global length
      length = len(a)
      b = []
      for line in a:
        b = b+line.split()+["\n"]
      return b
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  global word_to_id
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #print(len(word_to_id))
  return word_to_id


def _file_to_word_ids(filename):
  data = _read_words(filename)
  global word_to_id
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, is_training = True):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  #valid_path = os.path.join(data_path, "ptb.valid.txt")

  """
  global word_to_id
  word_to_id = eval(open("./cha_to_id.txt").read())
  if is_training == True:
    train_path = os.path.join(data_path, "Total_cha.txt")
    #word_to_id = _build_vocab(train_path)
    #f = open("./cha_to_id.txt","w")
    #f.write(str(word_to_id))
    #f.close()
    train_data = _file_to_word_ids(train_path)
  else:
    test_path = os.path.join(data_path, "test_char.txt")
    train_data = _file_to_word_ids(test_path)
  
  #vocabulary = len(word_to_id)
  ind = word_to_id["\n"]
  result = []
  tem = []
  co = 0
  global sequence_length
  for i in train_data:
    if i==ind:
      sequence_length.append(len(tem)-2)
      for temi in range(47-co):
        tem += [len(word_to_id)]
      result+=tem
      tem = []
      co = 0
      continue
    co += 1
    tem.append(i)
  """
  print("Getting Data...")
  global word_to_id
  word_to_id = eval(open("./cha_to_id.txt").read())
  global sequence_length
  if is_training == True:
    train_path = os.path.join(data_path, "corpus_cha.txt")
    #word_to_id = _build_vocab(train_path)
    #f = open("./cha_to_id.txt","w")
    #f.write(str(word_to_id))
    #f.close()
    train_data = array(open(train_path).read().replace("\n"," ").split(),dtype=int32)
    sequence_length = array( open(os.path.join(data_path, "corpus_cha_length.txt")).read().split() ,dtype = int32)
  else:
    test_path = os.path.join(data_path, "test_char.txt")
    train_data = array(open(test_path).read().replace("\n"," ").split(),dtype=int32)

  print("Getting Data Finish")
  return train_data


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  print("Producing batch...")
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    global sequence_length
    sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    # epoch_size  = tf.cond( x >0, lambda:x , lambda: tf.add(x,1))
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])

    seq_length = tf.strided_slice(sequence_length, [0],
                         [batch_size])
    seq_length.set_shape([batch_size])
    
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    print("Producing batch finish")
    return x, y, seq_length