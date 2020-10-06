import tensorflow as tf

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
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size, max_seq_len, idx_sos, idx_eos):
        super(seq2seq, self).__init__()        
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_size)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_size)

        # TODO: To clearly understand from_logits
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        self.max_seq_len = max_seq_len
        self.start_of_sentence = idx_sos
        self.end_token_idx = idx_eos

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

        dec_input = tf.expand_dims([self.start_of_sentence], 1)

        predict_tokens = list()
        for t in range(0, self.max_seq_len):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])

            if predict_token == self.end_token_idx:
                break

            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)
        
        return tf.stack(predict_tokens, axis=0).numpy()

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
