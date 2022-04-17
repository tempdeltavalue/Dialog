import tensorflow as tf
from tensorflow.keras.layers import Dense, add, LayerNormalization, Dropout
from transformer.components.multihead_attention import MultiHeadAttention
from transformer.components.positional_encoding import PositionalEncoding

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model),
                            name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model),
                                 name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None),
                                     name="look_ahead_mask")

    padding_mask = tf.keras.Input(shape=(1, 1, None),
                                  name='padding_mask')

    attention1 = MultiHeadAttention(d_model,
                                    num_heads,
                                    name="attention_1")(inputs={'query': inputs,
                                                                'key': inputs,
                                                                'value': inputs,
                                                                'mask': look_ahead_mask})

    add_attention = add([attention1, inputs])
    attention1 = LayerNormalization(epsilon=1e-6)(add_attention)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = Dropout(rate=dropout)(attention2)
    add_attention = add([attention2, attention1])
    attention2 = LayerNormalization(epsilon=1e-6)(add_attention)

    outputs = Dense(units=units, activation='relu')(attention2)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    add_attention = add([outputs, attention2])
    outputs = LayerNormalization(epsilon=1e-6)(add_attention)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],  #
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.keras.layers.Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model)
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])  # outputs, enc_outputs, look_ahead_mask, padding_mask

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],  # [inputs, enc_outputs, look_ahead_mask, padding_mask]
        outputs=outputs,
        name=name)