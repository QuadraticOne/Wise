from wise.networks.network import Network
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node, glorot_initialised_vars
import tensorflow as tf


class FeedforwardLayer(Network):
    """
    A basic feedforward layer as part of a neural network.
    """

    def __init__(self, name, session, input_shape, output_shape,
            activation=Activation.DEFAULT, input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [Int] -> (tf.Tensor -> String -> tf.Tensor)
            -> tf.Tensor? -> String? -> Network
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        
        self.input_node = input_node
        self.weights = None
        self.after_weights = None
        self.biases = None
        self.after_biases = None
        self.output_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        Initialise the tensorflow tensors that make up the layer.
        """
        if self.input_node is None:
            self.input_node = placeholder_node(self.extend_name('input_node'),
                self.input_shape, 1)
        self.weights = glorot_initialised_vars(self.extend_name('weights'),
            reverse(self.input_shape) + self.output_shape)
        self.after_weights = tf.tensordot(self.input_node, self.weights,
            len(self.input_shape), name=self.extend_name('after_weights'))
        self.biases = glorot_initialised_vars(self.extend_name('biases'), self.output_shape)
        self.after_biases = tf.add(self.after_weights, self.biases,
            name=self.extend_name('after_biases'))
        self.output_node = self.activation(self.after_biases, self.extend_name('output_node'))

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return [self.weights, self.biases]


def reverse(ls):
    return ls[::-1]
