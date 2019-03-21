from wise.util.tensors import placeholder_node
import tensorflow as tf


def regression_metrics(output_node_shape, output_node, name, variables=None):
    """
    [Int] -> tf.Tensor -> String -> [tf.Variable]?
        -> (TargetNode, LossNode, Optimiser)
    Create metrics - target node, loss node, and optimiser - for a
    regression model.
    """
    target_node = placeholder_node(
        name + ".target", output_node_shape, dynamic_dimensions=1
    )
    loss_node = tf.losses.mean_squared_error(target_node, output_node)
    optimiser = default_adam_optimiser(loss_node, name, variables=variables)
    return target_node, loss_node, optimiser


def classification_metrics(
    output_node_shape, output_node, name, variables=None, target=None
):
    """
    [Int] -> tf.Tensor -> String -> [tf.Variable]? ->
        (TargetNode, LossNode, Accuracy, Optimiser)
    Create metrics - target node, loss node, accuracy node, and optimiser -
    for a classification model.
    """
    target_node = (
        placeholder_node(name + ".target", output_node_shape, dynamic_dimensions=1)
        if target is None
        else target
    )
    loss_node = tf.losses.log_loss(target_node, output_node)
    accuracy_node = accuracy(output_node, target_node, name + ".accuracy")
    optimiser = default_adam_optimiser(loss_node, name, variables=variables)
    return target_node, loss_node, accuracy_node, optimiser


def beta_variational_metrics(
    input_node,
    reconstruction_node,
    means_node,
    stddevs_node,
    name,
    beta=1.0,
    variables=None,
    eps=1e-7,
):
    """
    tf.Tensor -> tf.Tensor -> tf.Tensor -> tf.Tensor -> String -> Float? ->
        [tf.Variable]? -> Float? -> (LossNode, LossNode, LossNode, Optimiser)
    Create three different loss nodes (reconstruction, KL-divergence,
    and composite) and an adam optimiser for a variational inference
    model. 
    """
    reconstruction_loss, _ = reconstruction_metrics(
        input_node,
        reconstruction_node,
        name + ".reconstruction_optimiser",
        variables=variables,
    )
    kl_loss, _ = variational_metrics(
        means_node, stddevs_node, name + ".variational_optimiser", variables=variables
    )
    composite_loss = reconstruction_loss + beta * kl_loss
    optimiser = default_adam_optimiser(
        composite_loss, name + ".composite_optimiser", variables=variables
    )
    return reconstruction_loss, kl_loss, composite_loss, optimiser


def reconstruction_metrics(input_node, reconstruction_node, name, variables=None):
    """
    tf.Tensor -> tf.Tensor -> String -> [tf.Variable]? -> (LossNode, Optimiser)
    Create reconstruction metrics for an autoencoder, which penalises the
    distance between an input and a reconstructed output in the vector
    space.
    """
    loss_node = tf.losses.mean_squared_error(input_node, reconstruction_node)
    optimiser = default_adam_optimiser(loss_node, name, variables=variables)
    return loss_node, optimiser


def variational_metrics(means_node, stddevs_node, name, variables=None, eps=1e-7):
    """
    tf.Tensor -> tf.Tensor -> String -> [tf.Variable]? -> Float?
        -> (LossNode, Optimiser)
    Create a KL-divergence loss for the Gaussian distributions described
    by the mean and standard deviation nodes provided.  Measures the KL-divergence
    from a standard Gaussian.
    """
    loss_node = tf.reduce_mean(
        0.5
        * tf.reduce_mean(
            tf.square(means_node)
            + tf.square(stddevs_node)
            - tf.log(tf.square(stddevs_node) + eps)
            - 1,
            1,
        )
    )
    optimiser = default_adam_optimiser(loss_node, name, variables=variables)
    return loss_node, optimiser


def default_adam_optimiser(loss_node, name, variables=None):
    """
    LossNode -> String -> Optimiser
    Create an adam optimiser with default learning parameters set to minimise
    the given loss node.
    """
    return tf.train.AdamOptimizer(name=name + ".adam_optimiser").minimize(
        loss_node, var_list=variables, name=name + ".minimise"
    )


def accuracy(prediction_node, target_node, name):
    """
    tf.Tensor -> tf.Tensor -> String -> tf.Tensor
    Create a node which measures the accuracy of a predictor as compared
    to a target node, where the prediction is considered correct if it is
    on the same side of 0.5 as the target.
    """
    prediction_positive = tf.less(0.5, prediction_node)
    target_positive = tf.less(0.5, target_node)
    correct = tf.equal(prediction_positive, target_positive)
    correct_floats = tf.cast(correct, tf.float32)
    return tf.reduce_mean(correct_floats, name=name)
