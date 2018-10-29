import tensorflow as tf


def l2_regularisation(network):
    """
    Network -> tf.Node
    Return a network that calculates the mean L2 loss of the
    parameters in the network.
    """
    variables = network.get_variables()
    variable_counts = network.feed([tf.size(variable) \
        for variable in variables], {})
    total_count = sum(variable_counts)
    individual_l2s = [(count / total_count) * tf.reduce_mean(tf.square(variable)) \
        for count, variable in zip(variable_counts, variables)]
    return tf.reduce_mean(individual_l2s, name=network.extend_name('l2'))
