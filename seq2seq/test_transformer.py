import numpy as np
import preprocess as prep
import tensorflow as tf
from model_transformer import Transformer

DIM_EMBEDDING = 30
NUM_UNITS = 100

if __name__ == "__main__":
    model_name = 'transformer'
    PATH = "models/" + model_name
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

    kargs = {'model_name': model_name,
    'num_layers': 2,
    'd_model': 512,
    'num_heads': 8,
    'dff': 2048,
    'input_vocab_size': len(word2index),
    'target_vocab_size': len(word2index),
    'maximum_position_encoding': prep.MAX_SEQUENCE,
    'end_token_idx': word2index[prep.EOS],
    'rate':0.1,         # drop out rate.
    'max_seq_len': prep.MAX_SEQUENCE,
    'idx_sos': word2index[prep.SOS]
    }

    # **kargs: keyword argument - receives dictionary as input.
    model = Transformer(**kargs)
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
