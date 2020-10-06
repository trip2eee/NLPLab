import numpy as np
import preprocess as prep
import tensorflow as tf
from model_seq2seq import seq2seq

DIM_EMBEDDING = 30
NUM_UNITS = 100

if __name__ == "__main__":
    PATH = "models/seq2seq"
    checkpoint_path = PATH + '/weights.h5'

    questions, answers = prep.load_data('kor2eng.csv')

    word2index, index2word = prep.load_dictionary()
    #print(word2index)
    #print(index2word)

    enc_inputs = prep.make_encoder_inputs(questions, word2index)
    #print(enc_inputs)

    print("make outputs")
    dec_inputs, dec_targets = prep.make_decoder_outputs(answers, word2index)
    #print(dec_inputs)
    #print(dec_targets)

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

    model = seq2seq(len(word2index), DIM_EMBEDDING, NUM_UNITS, NUM_UNITS, 1, prep.MAX_SEQUENCE, word2index[prep.SOS], word2index[prep.EOS])
    enc_input = enc_inputs_val[0]
    enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
    dec_output = model.inference(enc_input)

    #model = tf.keras.models.load_model(PATH)
    model.load_weights(checkpoint_path)

    """
    print("training set")
    for enc_input in enc_inputs:
        enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
        dec_output = model.inference(enc_input)

        ans = ""
        for i in dec_output:
            ans += index2word[i] + ' '
        print(ans)
    """
    
    print("validation set")
    for enc_input in enc_inputs_val:
        enc_input = np.reshape(enc_input, (1, enc_input.shape[0]))
        dec_output = model.inference(enc_input)

        ans = ""
        for i in dec_output:
            ans += index2word[i] + ' '
        print(ans)
