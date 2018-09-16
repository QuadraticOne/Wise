import tensorflow as tf


def placeholder_node(name, shape, dynamic_dimensions=0):
    """
    String -> [Int] -> Int? -> tf.Placeholder
    Create a placeholder node which accepts a float array of the given shape.
    The `dynamic_dimensions` argument should be set to 1 or more depending on
    the number of variable-size dimensions required at the start of the
    placeholder.
    """
    return tf.placeholder(tf.float32,
        shape=[None] * dynamic_dimensions + shape, name=name)


def glorot_initialised_vars(name, shape):
    """
    String -> [Int] -> tf.Variable
    Return a tensor of variables which are initialised according to the
    Glorot variable initialisation algorithm.
    """
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
        initializer=tf.glorot_normal_initializer())
