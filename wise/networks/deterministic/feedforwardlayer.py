from wise.networks.layer import Layer
from wise.networks.activation import Activation
from wise.networks.stochastic.noise import StochasticBinaryUnitLayer
from wise.util.tensors import glorot_initialised_vars
import tensorflow as tf


class FeedforwardLayer(Layer):
    """
    A basic feedforward layer as part of a neural network.
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
        weights_shape = reversed(self.input_shape) + self.output_shape
        self.weights = glorot_initialised_vars(self.extend_name('weights'),
            weights_shape)
        self.batch_normalised_weights = StochasticBinaryUnitLayer(
            self.extend_name('batch_normalised_weights'), self.get_session(),
            weights_shape, input_node=self.weights).get_output_node() \
            if self.batch_normalisation else self.weights

        self.after_weights = tf.tensordot(self.input_node, self.batch_normalised_weights,
            len(self.input_shape), name=self.extend_name('after_weights'))
        self.biases = glorot_initialised_vars(self.extend_name('biases'), self.output_shape)
        self.batch_normalised_biases = StochasticBinaryUnitLayer(
            self.extend_name('batch_normalised_biases'), self.get_session(),
            self.output_shape, input_node=self.biases).get_output_node() \
            if self.batch_normalisation else self.biases

        self.after_biases = tf.add(self.after_weights, self.batch_normalised_biases,
            name=self.extend_name('after_biases'))
        self.output_node = self.activation(self.after_biases,
            self.extend_name('output_node'))

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return [self.weights, self.biases]


def reversed(ls):
    return ls[::-1]
