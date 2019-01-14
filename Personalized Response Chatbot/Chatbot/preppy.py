'''
Tools to take a directory of txt files and convert them to TF records
'''
import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter

PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"
UNK = "<UNK>"
symbol_list = [PAD, START, EOS, UNK]

class Preppy():
    '''
    Class that converts text inputs to numpy arrays of ids.
    It assigns ids sequentially to the token on the fly.
    '''
    def __init__(self, tokenizer_fn, word_fequence=2):
        self.tokenizer = tokenizer_fn
        self.reverse_vocab = {}
        self.word_fequence = word_fequence

    def init_vocab(self):
        self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
        self.token_counter = Counter()  # Counts the token frequency
        self.vocab[PAD] = 0
        self.vocab[START] = 1
        self.vocab[EOS] = 2
        self.vocab[UNK] = 3
        self.next = 3

    def next_value(self):
        self.next += 1
        return self.next

    def prepare_vocab(self, sequence):
        self.sentence_to_id_list(sequence, False)

    def convert_token_to_id(self, token):
        self.token_counter[token] += 1
        return self.vocab[token]

    def vocab_filter(self):
        words = [word for (word, feq) in self.token_counter.items() if feq>self.word_fequence]
        words += symbol_list
        for word in list(self.vocab.keys()):
            if word not in words:
                self.vocab.pop(word)
                self.token_counter.pop(word)
        self.init_vocab()
        for word in words:
            self.vocab[word]

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
        #ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Add the tokens.
        #seq = ex.feature_lists.feature_list["length"]
        #seq.feature.add().int64_list.value.append(sequence_length)

        fl_tokens = ex.feature_lists.feature_list["sentence"]
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

    def convert_token_to_id_with_UNK(self, token):
        if token in self.vocab:
            return self.vocab[token]
        return self.vocab[UNK]

    # turn sentence into specific tokens like words or characters.
    def sentence_to_tokens(self, sent):
        return self.tokenizer(sent)

    # turn tokens into id list according to vocabulary
    def tokens_to_id_list(self, tokens, unk):
        if unk == True:
            return list(map(self.convert_token_to_id_with_UNK, tokens))
        else:
            return list(map(self.convert_token_to_id, tokens))

    # turn sentence into id list using sentence_to_tokens and tokens_to_id_list
    def sentence_to_id_list(self, sent, unk=True):
        tokens = self.sentence_to_tokens(sent)
        id_list = self.tokens_to_id_list(tokens, unk)
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
            
        }
        sequence_features = {
            "sentence": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"sentence": sequence_parsed["sentence"]}


class UserPreppy(Preppy):
    '''
    An extension of Preppy suited for ranking task of the table.
    It adds
    1) Storing the query、response、label、user index in the TFRecord
    '''
    def __init__(self, tokenizer_fn, word_fequence=5):
        super(UserPreppy, self).__init__(tokenizer_fn)
        vocab_file = open("./data/voca.pkl","rb")
        self.vocab = pickle.load(vocab_file)
        vocab_file.close()

    def sequence_to_tf_example(self, sequence, user, label):
        id_list = self.sentence_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        ex.context.feature["label"].int64_list.value.append(label)
        # user index, use to search user embedding matrix
        ex.context.feature["user"].int64_list.value.append(user)
        
        # add question feature
        sentence = ex.feature_lists.feature_list["sentence"]
        sentence.feature.add().int64_list.value.append(self.vocab[START])
        for token in id_list:
            sentence.feature.add().int64_list.value.append(token)
        sentence.feature.add().int64_list.value.append(self.vocab[EOS])

        return ex

    @staticmethod
    def parse(ex):
        # Explain to TF how to go from a serialized example back to tensors
        context_features = {
            "label": tf.FixedLenFeature([], dtype=tf.int64),
            "user": tf.FixedLenFeature([], dtype=tf.int64),
        }
        sequence_features = {
            "sentence": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"sentence": sequence_parsed["sentence"],
                "user": context_parsed["user"], "label": context_parsed["label"]}



class RankPreppy(Preppy):
    '''
    An extension of Preppy suited for ranking task of the table.
    It adds
    1) Storing the query、response、label、user index in the TFRecord
    '''
    def __init__(self, tokenizer_fn, word_fequence=5):
        super(RankPreppy, self).__init__(tokenizer_fn)
        vocab_file = open("./data/voca.pkl","rb")
        self.vocab = pickle.load(vocab_file)
        vocab_file.close()

    def sequence_to_tf_example(self, query, response, user, label):
        id_list_q = self.sentence_to_id_list(query)
        id_list_a = self.sentence_to_id_list(response)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        
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
        # Explain to TF how to go from a serialized example back to tensors
        context_features = {
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
        return {"query": sequence_parsed["query"],
                "response": sequence_parsed["response"],
                "user": context_parsed["user"], 
                "label": context_parsed["label"]}
