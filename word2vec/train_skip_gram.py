"""
@fn     train_skip_gram.py
@brief  Skip-gram training code.
@author trip2eee@gmail.com
@date   October 02, 2020
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle

class SkipGram(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()

        self.embd_in = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.embd_out_w = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.embd_out_bias = layers.Embedding(input_dim=vocab_size, output_dim=1)
        self.dot = layers.Dot(axes=(1, 2))
        self.activation = layers.Activation('sigmoid')

        self.vocab_size = vocab_size

    def call(self, x):

        idx_in = x[:,0]
        idx_out = x[:,1:3]

        x = self.embd_in(idx_in)

        w_out = self.embd_out_w(idx_out)
        b_out = self.embd_out_bias(idx_out)
        b_out = tf.reshape(b_out, (-1, 2))

        x = self.dot([x, w_out])
        x = tf.math.add(x, b_out)
        x = self.activation(x)
      
        return x

if __name__ == "__main__":
    model_path = "models/SkipGram"        
    EPOCHS = 10000

    corpus = [
    'I am a boy.',
    'I am a girl.',
    'I am a man.',
    'I am a woman.',
    'You are a boy.',
    'You are a girl.',
    'You are a man.',
    'You are a woman.']

    for i in range(0, len(corpus)):
        corpus[i] = corpus[i].lower()
        corpus[i] = corpus[i].replace('.', ' .')

    # Default tokenizer select tokens of 2 or more alphanumeric characters.
    # To make the tokenizer not to ignore words with only one letter, a new tokenizer has to be defuled.
    count_vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
    count_vectorizer.fit(corpus)

    word2index = count_vectorizer.vocabulary_

    index2word = dict([ (word2index[key], key) for key in word2index] )

    print(word2index)
    print(index2word)
    
    with open(model_path + '/word2index.bin', 'wb') as f:
        pickle.dump(word2index, f)

    with open(model_path + '/index2word.bin', 'wb') as f:
        pickle.dump(index2word, f)


    vocab_size = len(word2index)
    EOS = word2index['.']       # End of Sentence

    pdf = np.zeros(vocab_size).astype(np.float32)

    # convert workd to index.
    corpus_index = []
    for s in corpus:
        index = []
        tokens = s.split()
        for t in tokens:            
            i = word2index[t]
            index.append(i)

            pdf[i] += 1.0
        corpus_index.append(index)

    pdf = pdf / np.sum(pdf)

    print(corpus_index)

    model = SkipGram(vocab_size=vocab_size, embedding_dim=2)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_losss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')


    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    for epoch in range(EPOCHS):
        
        # create batch
        for c in corpus_index:
            words = []
            labels = []
            
            for i in range(0, len(c)):
                idx0 = i-1
                idx1 = i+1
                
                if idx0 >= 0:
                    c0 = c[idx0]
                else:
                    c0 = EOS

                if idx1 < len(c):
                    c1 = c[idx1]
                else:
                    c1 = EOS

                # positive
                w_pos = [c[i], c0, c1]
                words.append(w_pos)
                labels.append([1, 1])

                # negative
                # select random words.
                idx_out = np.random.choice(vocab_size, size=2, p=pdf)

                # determine target output.
                if idx_out[0] in [c0, c1]:
                    t0 = 1
                else:
                    t0 = 0
                
                if idx_out[1] in [c0, c1]:
                    t1 = 1
                else:
                    t1 = 0

                w_neg = [c[i], idx_out[0], idx_out[1]]
                words.append(w_neg)
                labels.append([t0, t1])

            words = np.array(words).astype(np.int32)
            labels = np.array(labels).astype(np.int32)
            

            train_step(words, labels)

        template = 'Epoch: {}, loss: {:.6f}, accuracy: {:.2f}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100))

    
    model.save(model_path)
 

    




