'''
Tools to take a directory of txt files and convert them to TF records
'''
from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf
PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"

class Preppy():
    '''
    Class that converts text inputs to numpy arrays of ids.
    It assigns ids sequentially to the token on the fly.
    '''
    def __init__(self, tokenizer_fn):
        self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
        self.token_counter = Counter()  # Counts the token frequency
        self.vocab[PAD] = 0
        self.vocab[START] = 1
        self.vocab[EOS] = 2
        self.next = 2
        self.tokenizer = tokenizer_fn
        self.reverse_vocab = {}

    def next_value(self):
        self.next += 1
        return self.next

    def sequence_to_tf_example(self, sequence):
        '''
        Gets a sequence (a text like "hello how are you") and returns a a SequenceExample
        :param sequence: Some text
        :return: A A sequence exmaple
        '''
        # Convert the text to a list of ids
        id_list = self.sentence_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(id_list) + 2  # For start and end
        # Add the context feature, here we just need length
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Add the tokens.
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        # Prepend with start token
        fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list:
            # Add those tokens (features) one by one
            fl_tokens.feature.add().int64_list.value.append(token)
        # apend  with end token
        fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
        return ex

    def ids_to_string(self, tokens, length=None):
        string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
        return string

    def convert_token_to_id(self, token):
        '''
        Gets a token, looks it up in the vocabulary. If it doesn't exist in the vocab, it gets added to id with an id
        Then we return the id
        :param token:
        :return: the token id in the vocab
        '''
        self.token_counter[token] += 1
        return self.vocab[token]

    # turn sentence into specific tokens like words or characters.
    def sentence_to_tokens(self, sent):
        return self.tokenizer(sent)

    # turn tokens into id list according to vocabulary
    def tokens_to_id_list(self, tokens):
        return list(map(self.convert_token_to_id, tokens))

    # turn sentence into id list using sentence_to_tokens and tokens_to_id_list
    def sentence_to_id_list(self, sent):
        tokens = self.sentence_to_tokens(sent)
        id_list = self.tokens_to_id_list(tokens)
        return id_list

    def sentence_to_numpy_array(self, sent):
        id_list = self.sentence_to_id_list(sent)
        return np.array(id_list)

    def update_reverse_vocab(self):
        self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

    def id_list_to_text(self, id_list):
        tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
        return tokens

    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go from a serialized example back to tensors
        :param ex:
        :return: A dictionary of tensors, in this case {seq: The sequence, length: The length of the sequence}
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq": sequence_parsed["tokens"],
                "length": context_parsed["length"]}


class RankPreppy(Preppy):
    '''
    An extension of Preppy suited for ranking task of the table.
    It adds
    1) Storing the query、response、label、user index in the TFRecord
    '''
    def __init__(self, tokenizer_fn):
        super(RankPreppy, self).__init__(tokenizer_fn)

    def sequence_to_tf_example(self, sequence_q, sequence_a, user, label):
        id_list_q = self.sentence_to_id_list(sequence_q)
        id_list_a = self.sentence_to_id_list(sequence_a)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length_q = len(sequence_q)
        sequence_length_a = len(sequence_a)
        
        # query length
        ex.context.feature["length_q"].int64_list.value.append(sequence_length_q + 2)
        # response length
        ex.context.feature["length_a"].int64_list.value.append(sequence_length_a + 2)
        # label, the response is appropriate to the query,(0:inappropriate, 1:appropriate)
        ex.context.feature["label"].int64_list.value.append(label)
        # user index, use to search user embedding matrix
        ex.context.feature["user"].int64_list.value.append(user)
        
        # add question feature
        query = ex.feature_lists.feature_list["query"]
        query.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list_q:
            query.feature.add().int64_list.value.append(token)
        query.feature.add().int64_list.value.append(self.vocab[EOS])

        # add response feature
        response = ex.feature_lists.feature_list["response"]
        response.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list_a:
            response.feature.add().int64_list.value.append(token)
        response.feature.add().int64_list.value.append(self.vocab[EOS])
        
        return ex

    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go from a serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length_q": tf.FixedLenFeature([], dtype=tf.int64),
            "length_a": tf.FixedLenFeature([], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64),
            "user": tf.FixedLenFeature([], dtype=tf.int64),
        }
        sequence_features = {
            "query": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "response": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"query": sequence_parsed["query"], "length_q": context_parsed["length_q"],
                "response": sequence_parsed["response"], "length_a": context_parsed["length_a"],
                "user": context_parsed["user"], "label": context_parsed["label"]}
