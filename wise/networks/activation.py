import tensorflow as tf


def identity(x, _):
    """
    Object -> String -> Object
    """
    return x


class Activation:
    """
    Various constants for easy use of activation functions.
    """

    SIGMOID = tf.nn.sigmoid
    RELU = tf.nn.relu
    LEAKY_RELU = tf.nn.leaky_relu
    TANH = tf.nn.tanh
    
    DEFAULT = tf.nn.sigmoid
    IDENTITY = identity
