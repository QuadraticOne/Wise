from wise.networks.network import Network
from wise.networks.deterministic.feedforwardlayer import FeedforwardLayer
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node
import tensorflow as tf


class FeedforwardNetwork(Network):
    """
    Feedforward neural netowrk, consisting of a number of layers
    of fully connected neurons.
    """

    def __init__(self, name, session, input_shape, layer_shapes, activations=
            Activation.all(Activation.DEFAULT), input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [[Int]] ->
            (Int -> [(tf.Tensor -> String -> tf.Tensor)]) -> tf.Tensor? -> String? ->
            FeedforwardNetwork
        
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.layer_shapes = layer_shapes
        self.layer_activations = activations(len(self.layer_shapes))
        
        self.input_node = input_node
        self.layers = []
        self.output_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        if self.input_node is None:
            self.input_node = placeholder_node(self.extend_name('input_node'),
                self.input_shape, 1)

        previous_layer_output_node = self.input_node
        previous_layer_shape = self.input_shape
        self.layers = []

        for shape, activation in zip(self.layer_shapes, self.layer_activations):
            new_layer = FeedforwardLayer(
                name=self.extend_name('layer_' + str(len(self.layers))),
                session=self.get_session(),
                input_shape=previous_layer_shape,
                output_shape=shape,
                activation=activation,
                input_node=previous_layer_output_node
            )
            self.layers.append(new_layer)
            previous_layer_output_node = new_layer.output_node
            previous_layer_shape = shape[:]
        self.output_node = previous_layer_output_node

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        output = []
        for layer in self.layers:
            output += layer.get_variables()
        return output
