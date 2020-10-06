import re
import pandas as pd
from konlpy.tag import Okt
import pickle

#FILTERS = "([~.,!?\"':;)(])"
FILTERS = "([.,\"':;)(])"
MAX_SEQUENCE = 10

PAD = '<PAD>' # PAD: padding.    
SOS = '<SOS>' # SOS: Start of sentence.
EOS = '<EOS>' # EOS: End of sentence.
UNK = '<UNK>' # UNK: word that does not exists in dictionary.

def load_data(path):
    # load data.
    data_set = pd.read_csv(path, header=0)
    data_set.describe()

    questions = data_set['KOR']
    answers = data_set['ENG']
    
    return questions, answers

def tokenize(data):
    okt = Okt()

    CHANGE_FILTER = re.compile(FILTERS)
    
    words = []

    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        sentence = sentence.lower()

        #for word in sentence.split():
        for word in okt.morphs(sentence):
            words.append(word)
    return [word for word in words]


def make_dictionary(data):
    
    # words list starts with markers.    
    words = [PAD, SOS, EOS, UNK]
    words.extend(tokenize(data))

    dictionary = []
    for word in words:
        if word not in dictionary:
            dictionary.append(word)

    print(dictionary)

    word2index = {word:idx for idx, word in enumerate(dictionary)}
    index2word = {idx:word for idx, word in enumerate(dictionary)}

    with open('word2index.bin', 'wb') as f:
        pickle.dump(word2index, f)

    with open('index2word.bin', 'wb') as f:
        pickle.dump(index2word, f)

    return word2index, index2word

def load_dictionary():
    with open('word2index.bin', 'rb') as f:
        word2index = pickle.load(f)

    with open('index2word.bin', 'rb') as f:
        index2word = pickle.load(f)

    return word2index, index2word

def make_encoder_inputs(sentences, word2index):
    #encoder input: i0 i1 i2 <PAD> <PAD>...

    okt = Okt()
    sentences_idx = []    
    CHANGE_FILTER = re.compile(FILTERS)

    for stc in sentences:
        idx = []
        stc = re.sub(CHANGE_FILTER, "", stc)
        stc = stc.lower()

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
        stc = stc.lower()
        
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