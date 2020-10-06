import os
import re
import json

import numpy as np


import tensorflow as tf
import numpy as np
import preprocess as prep
from model_seq2seq import seq2seq

DIM_EMBEDDING = 30
NUM_UNITS = 100


if __name__ == "__main__":
    
    questions, answers = prep.load_data('kor2eng.csv')

    data = []    
    data.extend(questions)
    data.extend(answers)
    
    word2index, index2word = prep.make_dictionary(data)

    enc_inputs = prep.make_encoder_inputs(questions, word2index)

    print("make outputs")
    dec_inputs, dec_targets = prep.make_decoder_outputs(answers, word2index)

    # Validation set.
    questions_val, answers_val = prep.load_data('kor2eng_val.csv')
    
    enc_inputs_val = prep.make_encoder_inputs(questions_val, word2index)
    dec_inputs_val, dec_targets_val = prep.make_decoder_outputs(answers_val, word2index)
    
    enc_inputs = np.array(enc_inputs)
    dec_inputs = np.array(dec_inputs)
    dec_targets = np.array(dec_targets)

    enc_inputs_val = np.array(enc_inputs_val)
    dec_inputs_val = np.array(dec_inputs_val)
    dec_targets_val = np.array(dec_targets_val)


    optimizer = tf.keras.optimizers.Adam()
    
    PATH = "models/seq2seq"
    if not (os.path.isdir(PATH)):
        os.makedirs(os.path.join(PATH))

    checkpoint_path = PATH + '/weights.h5'

    model = seq2seq(len(word2index), DIM_EMBEDDING, NUM_UNITS, NUM_UNITS, 1, prep.MAX_SEQUENCE, word2index[prep.SOS], word2index[prep.EOS])

    if True:
        model.compile(loss=model.custom_loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[model.custom_accuracy])        

        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_custom_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_custom_accuracy', min_delta=0.001, patience=50)

        history = model.fit([enc_inputs, dec_inputs], dec_targets, batch_size=1, epochs=300, validation_data=([enc_inputs_val, dec_inputs_val], dec_targets_val), callbacks=[cp_callback])
    
    # Test the trained model.
    model.load_weights(checkpoint_path)

    print("training set")
    for enc_input in enc_inputs:
        enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
        dec_output = model.inference(enc_input)

        ans = ""
        for i in dec_output:
            ans += index2word[i] + ' '
        print(ans)

    print("validation set")
    for enc_input in enc_inputs_val:
        enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
        dec_output = model.inference(enc_input)

        ans = ""
        for i in dec_output:
            ans += index2word[i] + ' '
        print(ans)



