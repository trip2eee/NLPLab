import os
import re
import json

import numpy as np
import pandas as pd

from konlpy.tag import Okt

import tensorflow as tf
import numpy as np


MAX_SEQUENCE = 10
#FILTERS = "([~.,!?\"':;)(])"
FILTERS = "([~.,\"':;)(])"
PAD = '<PAD>' # PAD: padding.    
SOS = '<SOS>' # SOS: Start of sentence.
EOS = '<EOS>' # EOS: End of sentence.
UNK = '<UNK>' # UNK: word that does not exists in dictionary.

def load_data(path):
    # load data.
    data_set = pd.read_csv(path, header=0)
    data_set.describe()

    questions = data_set['Q']
    answers = data_set['A']
    
    return questions, answers

def tokenize(data):
    okt = Okt()

    CHANGE_FILTER = re.compile(FILTERS)
    
    words = []

    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        
        #for word in sentence.split():
        for word in okt.morphs(sentence):
            words.append(word)
    return [word for word in words]


def make_dictionary(words):
    dictionary = []
    for word in words:
        if word not in dictionary:
            dictionary.append(word)

    print(dictionary)

    word2index = {word:idx for idx, word in enumerate(dictionary)}
    index2word = {idx:word for idx, word in enumerate(dictionary)}

    return word2index, index2word


def make_encoder_inputs(sentences, word2index):
    #encoder input: i0 i1 i2 <PAD> <PAD>...

    okt = Okt()
    sentences_idx = []    
    CHANGE_FILTER = re.compile(FILTERS)

    for stc in sentences:
        idx = []
        stc = re.sub(CHANGE_FILTER, "", stc)

        for w in okt.morphs(stc):
            if w in word2index:
                idx.extend([word2index[w]])
            else:
                idx.extend([word2index[UNK]])
        
        idx += ((MAX_SEQUENCE - len(idx)) * [word2index[PAD]])
        sentences_idx.append(idx)
    
    return sentences_idx


def make_decoder_outputs(sentences, word2index):
    #decoder input: <SOS> i0 i1 i2 <PAD> <PAD>...
    #decoder target: i0 i1 i2 <EOS> <PAD> <PAD>...

    okt = Okt()    
    CHANGE_FILTER = re.compile(FILTERS)

    dec_inputs = []
    dec_targets = []

    for stc in sentences:
        
        idx_in = [word2index[SOS]]
        idx_target = []

        stc = re.sub(CHANGE_FILTER, "", stc)
        for w in okt.morphs(stc):
            if w in word2index:
                idx_in.extend([word2index[w]])
                idx_target.extend([word2index[w]])
            else:
                idx_in.extend([word2index[UNK]])
                idx_target.extend([word2index[UNK]])
        
        idx_target.extend([word2index[EOS]])

        idx_in += ((MAX_SEQUENCE - len(idx_in)) * [word2index[PAD]])
        idx_target += ((MAX_SEQUENCE - len(idx_target)) * [word2index[PAD]])

        dec_inputs.append(idx_in)
        dec_targets.append(idx_target)
    
    return dec_inputs, dec_targets

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,        # dimensionality of the output space.
                                       return_sequences=True, # Whether to return the last output in the output sequence, or the full sequence.
                                       return_state=True,     # Whether to return the last state in addition to the output
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self, inputs):
        return tf.zeros((tf.shape(inputs)[0], self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_mean(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

class seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size, end_token_idx=2):
        super(seq2seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_size)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_size)

    def call(self, x):
        inputs, targets = x

        enc_hidden = self.encoder.initialize_hidden_state(inputs)
        enc_output, enc_hidden = self.encoder(inputs, enc_hidden)

        dec_hidden = enc_hidden
        
        predict_tokens = list()
        for t in range(0, targets.shape[1]):
            dec_input = tf.dtypes.cast(tf.expand_dims(targets[:,t], 1), tf.float32)
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))

        return tf.stack(predict_tokens, axis=1)

    
    def inference(self, x):
        inputs = x

        enc_hidden = self.encoder.initialize_hidden_state(inputs)
        enc_output, enc_hidden = self.encoder(inputs, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([word2index[SOS]], 1)

        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])

            if predict_token == self.end_token_idx:
                break

            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)
        
        return tf.stack(predict_tokens, axis=0).numpy()


if __name__ == "__main__":
    
    questions, answers = load_data('data_set.csv')

    print("Questions")
    print(questions)

    print("Answer")
    print(answers)

    data = []    
    data.extend(questions)
    data.extend(answers)

    words = tokenize(data)

    # add markers.    
    MARKERS = [PAD, SOS, EOS, UNK]

    MARKERS.extend(words)
    words = MARKERS
    print(words)

    word2index, index2word = make_dictionary(words)

    print(word2index)
    print(index2word)

    enc_inputs = make_encoder_inputs(questions, word2index)
    print(enc_inputs)

    print("make outputs")
    dec_inputs, dec_targets = make_decoder_outputs(answers, word2index)
    print(dec_inputs)
    print(dec_targets)

    optimizer = tf.keras.optimizers.Adam()
    # TODO: To clearly understand from_logits
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def loss(real, pred):
        # find values which are not <PAD>
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def accuracy(real, pred):
        # find values which are not <PAD>
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
        pred *= mask
        acc = train_accuracy(real, pred)
        
        return tf.reduce_mean(acc)


    model = seq2seq(len(word2index), 10, 100, 100, 1, word2index[EOS])
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[accuracy])

    PATH = "models/seq2seq"
    if not (os.path.isdir(PATH)):
        os.makedirs(os.path.join(PATH))

    checkpoint_path = PATH + '/weights.h5'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    enc_inputs = np.array(enc_inputs)
    dec_inputs = np.array(dec_inputs)
    dec_targets = np.array(dec_targets)

    print('input data shapes')
    print(enc_inputs.shape)
    print(dec_inputs.shape)
    print(dec_targets.shape)

    history = model.fit([enc_inputs, dec_inputs], dec_targets, batch_size=1, epochs=200, callbacks=[cp_callback])

    """
    dec_outputs = model.inference(enc_inputs)

    for idx in dec_outputs:
        ans = [index2word[i] + ' ' for i in idx]
        print(ans)
    """
    
    for enc_input in enc_inputs:

        enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
        dec_output = model.inference(enc_input)

        ans = ""
        for i in dec_output:
            ans += index2word[i] + ' '
        print(ans)
    