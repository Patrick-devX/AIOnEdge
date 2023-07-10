import tensorflow as tf


def mlp (input_, conf_file):
    """
    Main Unit Arcgutecture
    :param input_: (tensor) Input Layer
    :param conf_file: (json) Configuration file
    :return:
    """
    x = tf.keras.layers.Dense(conf_file['mlp_dim'], activation='glu')(input_)
    x = tf.keras.layers.Dropout(conf_file['dropout_rate'])(x)
    x = tf.keras.layers.Dense(conf_file['hidden_dim'])(x)
    x = tf.keras.layers.Dropout(conf_file['dropout_rate'])(x)
    return x

def transformer_encoder(input_, conf_file):
    skip_input = input_
    x = tf.keras.layers.LayerNormalization()(input_)
    x = tf.keras.layers.MultiHeadAttention(num_heads=conf_file['num_heads'], key_dim=conf_file['hidden_dim'])(x, x)
    x = tf.keras.layers.Add()[x, skip_input]

    skip_input_attention = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = mlp(x, conf_file=conf_file)
    x = tf.keras.layers.Add()[x, skip_input_attention]
    return x

def convolution_block(input_, num_filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input_)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.ReLU(x)
    return x

def deconv_block(input_, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=2, padding='same', strides=2)(input_)
    return x

def build_unetr_2d

if __name__ == '__main__':
    conf_file = {}
    conf_file['image_size'] = 256
    conf_file['num_layers'] = 12
    conf_file['hidden_dim'] = 768
    conf_file['mlp_dim'] = 3072
    conf_file['num_heads'] = 12
    conf_file['dropout_rate'] = .1
    conf_file['num_patches'] = 256
    conf_file['patch_size'] = 16
    conf_file['num_channels'] = 3

