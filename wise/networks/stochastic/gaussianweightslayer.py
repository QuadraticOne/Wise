from wise.networks.layer import Layer
from wise.networks.activation import Activation
from wise.networks.deterministic.feedforwardlayer import FeedforwardLayer
from wise.util.tensors import glorot_initialised_vars
import tensorflow as tf


class GaussianWeightsLayer(Layer):
    """
    A neural network layer which, instead of having deterministic weights,
    samples its weights from independently defined Gaussian distributions.
    """

    def __init__(self, name, session, input_shape, output_shape,
            activation=Activation.DEFAULT, input_node=None,
            batch_normalisation=False, save_location=None):
        """
        String -> tf.Session -> [Int] -> [Int] -> (tf.Tensor -> String -> tf.Tensor)
            -> tf.Tensor? -> Bool? -> String? -> Network
        """
        super().__init__(name, session, input_shape, output_shape,
            input_node, save_location)

        self.activation = activation

        self.weight_means = None
        self.weight_stddevs = None
        self.weight_noise = None
        self.weights = None
        self.after_weights = None
        self.biases = None
        self.after_biases = None

        self.batch_normalisation = batch_normalisation
        self.batch_normalised_weights = None
        self.batch_normalised_biases = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        Initialise the tensorflow tensors that make up the layer.
        """
        weight_shape = reversed(self.input_shape) + self.output_shape
        self.weight_means = glorot_initialised_vars(
            self.extend_name('weight_means'), weight_shape)
        self.weight_stddevs = glorot_initialised_vars(
            self.extend_name('weight_stddevs'), weight_shape)
        self.weight_noise = tf.random_normal(weight_shape, 0, 1,
            name=self.extend_name('noise'))
        self.weights = tf.add(self.weight_means, tf.multiply(self.weight_noise,
            self.weight_stddevs), name=self.extend_name('weights'))
        self.batch_normalised_weights = StochasticBinaryUnitLayer(
            self.extend_name('batch_normalised_weights'), self.get_session(),
            weight_shape, input_node=self.weights).get_output_node() \
            if self.batch_normalisation else self.weights

        self.after_weights = tf.tensordot(self.input_node, self.weights,
            len(self.input_shape), name=self.extend_name('after_weights'))
        self.biases = glorot_initialised_vars(self.extend_name('biases'), self.output_shape)
        self.batch_normalised_biases = StochasticBinaryUnitLayer(
            self.extend_name('batch_normalised_biases'), self.get_session(),
            self.output_shape, input_node=self.biases).get_output_node() \
            if self.batch_normalisation else self.biases
        self.after_biases = tf.add(self.after_weights, self.biases,
            name=self.extend_name('after_biases'))
        self.output_node = self.activation(self.after_biases,
            self.extend_name('output_node'))

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return [self.weight_means, self.weight_stddevs, self.biases]


def reversed(ls):
    return ls[::-1]

