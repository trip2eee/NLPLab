"""
@fn     test_skip_gram.py
@brief  Skip-gram test code.
@author trip2eee@gmail.com
@date   October 02, 2020
"""

import tensorflow as tf
from tensorflow.keras import Model    
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_path = 'models/SkipGram'
    model = tf.keras.models.load_model(model_path)

    with open(model_path + '/word2index.bin', 'rb') as f:
        word2index = pickle.load(f)

    with open(model_path + '/index2word.bin', 'rb') as f:
        index2word = pickle.load(f)

    
    embd_in = model.get_weights()[0]

    print(embd_in)

    plt.figure('word2vec')

    for key in word2index:
        vec = embd_in[word2index[key]]

        plt.scatter(vec[0], vec[1])
        plt.text(vec[0], vec[1], key)
    
    plt.savefig('skip_gram_result.png')
    plt.show()    



