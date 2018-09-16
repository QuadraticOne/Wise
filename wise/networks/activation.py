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
    LEAKY_RELU = lambda t, s: tf.nn.leaky_relu(t, name=s)
    TANH = tf.nn.tanh
    
    DEFAULT = tf.nn.sigmoid
    IDENTITY = identity

    @staticmethod
    def all(activation):
        """
        (tf.Tensor -> String -> tf.Tensor) -> (Int -> [(tf.Tensor -> String -> tf.Tensor)])
        Replicates the given activation function once for each layer.
        """
        def apply(n_layers):
            return [activation] * n_layers
        return apply

    @staticmethod
    def all_except_last(internal, output):
        """
        (tf.Tensor -> String -> tf.Tensor) -> (tf.Tensor -> String -> tf.Tensor)
            -> (Int -> [(tf.Tensor -> String -> tf.Tensor)])
        Repeat the same activation function for all the internal layers of the network,
        and then switch to a different one for the output layer only.
        """
        def apply(n_layers):
            return [internal] * (n_layers - 1) + [output]
        return apply
