"""
@file  model_transformer.py
@brief Transformer model implementation.
@reference https://github.com/NLP-kr/tensorflow-ml-nlp-tf2
"""

import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    # q, k, v: (batch_size, num_heads, seq_len, d_model / num_heads)
    # matmul_qk: (batch_size, num_heads, seq_len, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)        # assign very small number to the upper triangular part.

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)     # weights for the masked positions = 0.
    output = tf.matmul(attention_weights, v)        # (batch_size, num_heads, seq_len, d_model / num_heads)

    return output, attention_weights

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):

    # tf.linalg.band_part: create a matrix in which the lower triangular part is filled with 1s (num_lower < 0: Copy entire lower triangular).    
    mask = 1 - tf.linalg.band_part(input=tf.ones((size, size)), num_lower=-1, num_upper=0)

    # mask = [[0 1 1 1]
    #         [0 0 1 1]
    #         [0 0 0 1]
    #         [0 0 0 0]]
    return mask

def create_masks(input, target):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(input)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def position_wise_feed_forward_network(d_model, dff):
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

    return net

def get_angles(pos, i, d_model):
    # pos / (10000^(2*i/d_model))
    angle_rates = pos / np.power(10000, (2 * i) / np.float32(d_model))
    return angle_rates

def positional_encoding(position, d_model):    
    # np.arange(2) -> [0, 1], np.arange(2)[:,np.newaxis] -> [[0], [1]]
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis,:],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # angle_rads.shape:   (seq_len, d_model)
    # pos_encoding.shape: (batch_size, seq_len, d_model)
    # The ellipsis syntax(...) selects full any remaining unspecified dimensions.
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    # **kargs: keyword argument - receives dictionary as input.
    def __init__(self, **kargs):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = kargs['num_heads']
        self.d_model = kargs['d_model']

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads     # a // b == int(a/b)

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    # This method splits input tensor of shape (batch, seqeunce, feature) into (batch, head, sequence, feature)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)      # (batch_size, seq_len, d_model)
        k = self.wk(k)      # (batch_size, seq_len, d_model)
        v = self.wv(v)      # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


"""
Encoder layer.
"""
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(**kargs)

        self.ffn = position_wise_feed_forward_network(kargs['d_model'], kargs['dff'])

        # Normalization layers apply a transformation that maintains the mean activation within each example 
        # close to 0 and the activation standard deviation close to 1.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):
        # value, key, query: x
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

"""
Encoder - a stack of N identical encoder layers.
"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Encoder, self).__init__()

        self.d_model = kargs['d_model']
        self.num_layers = kargs['num_layers']       # The number of layers in the encoder stack (N in the paper).

        self.embedding = tf.keras.layers.Embedding(kargs['input_vocab_size'], self.d_model)
        self.pos_encoding = positional_encoding(kargs['maximum_position_encoding'], self.d_model)

        self.enc_layers = [EncoderLayer(**kargs) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]      # shape: (batch_size, seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x

"""
Decoder layer
"""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(**kargs)
        self.mha2 = MultiHeadAttention(**kargs)

        self.ffn = position_wise_feed_forward_network(kargs['d_model'], kargs['dff'])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout3 = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        # value, key, query: x
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        # value, key: enc_output, query: out1
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

"""
Decoder - a stack of N identical decoder layers.
"""
class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Decoder, self).__init__()

        self.d_model = kargs['d_model']
        self.num_layers = kargs['num_layers']       # The number of layers in the encoder stack (N in the paper).

        self.embedding = tf.keras.layers.Embedding(kargs['target_vocab_size'], self.d_model)
        self.pos_encoding = positional_encoding(kargs['maximum_position_encoding'], self.d_model)

        self.dec_layers = [DecoderLayer(**kargs) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
        
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        return x, attention_weights
    
class Transformer(tf.keras.Model):
    def __init__(self, **kargs):
        super(Transformer, self).__init__()
        self.end_token_idx = kargs['end_token_idx']

        self.encoder = Encoder(**kargs)
        self.decoder = Decoder(**kargs)

        self.final_layer = tf.keras.layers.Dense(kargs['target_vocab_size'])        # activation: linear - not normalized.

        self.max_seq_len = kargs['max_seq_len']
        self.start_of_sentence = kargs['idx_sos']

        # from_logits=True: The output is not normalized. Softmax will be applied in the loss function.
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def call(self, x):
        input, target = x

        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(input, target)
        enc_output = self.encoder(input, enc_padding_mask)
        dec_output, _ = self.decoder(target, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output
    
    def inference(self, x):
        input = x
        target = tf.expand_dims([self.start_of_sentence], 0)

        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(input, target)
        enc_output = self.encoder(input, enc_padding_mask)

        predict_tokens = list()
        for t in range(0, self.max_seq_len):
            dec_output, _ = self.decoder(target, enc_output, look_ahead_mask, dec_padding_mask)
            final_output = self.final_layer(dec_output)
            outputs = tf.argmax(final_output, -1).numpy()       # retrieve the word with the highest probability. The output don't have to be normalized.
            pred_token = outputs[0][-1]

            if pred_token == self.end_token_idx:
                break
        
            predict_tokens.append(pred_token)
            target = tf.expand_dims([self.start_of_sentence] + predict_tokens, 0)
            _, look_ahead_mask, dec_padding_mask = create_masks(input, target)

        return predict_tokens

    
    def custom_loss(self, real, pred):
        # find values which are not <PAD>
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def custom_accuracy(self, real, pred):
        # find values which are not <PAD>
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
        pred *= mask
        acc = self.train_accuracy(real, pred)
        
        return tf.reduce_mean(acc)









