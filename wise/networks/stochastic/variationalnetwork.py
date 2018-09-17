from wise.networks.network import Network
from wise.networks.stochastic.variationallayer import VariationalLayer
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node

class VariationalNetwork(Network):
    """
    Similar to a standard feedforward network but with a variational output
    layer.
    """

    def __init__(self, name, session, input_shape, layer_shapes,
            internal_activations=Activation.all(Activation.DEFAULT),
            means_activation=Activation.TANH, stddevs_activation=Activation.SIGMOID,
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [[Int]] -> (Int -> [(tf.Tensor -> String -> tf.Tensor)])?
            -> (tf.Tensor -> String -> tf.Tensor)? -> (tf.Tensor -> String -> tf.Tensor)?
            -> tf.Tensor? -> String? -> VariationalNetwork
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.layer_shapes = layer_shapes
        self.internal_activations = internal_activations
        self.means_activation = means_activation
        self.stddevs_activation = stddevs_activation

        self.input_node = input_node
        self.internal_network = None
        self.output_layer = None
        self.means_output_node = None
        self.stddevs_output_node = None
        self.sample_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        if self.input_node is None:
            self.input_node = placeholder_node(self.extend_name('input_node'),
                self.input_shape, dynamic_dimensions=1)
        
        self.internal_network = FeedforwardNetwork(
            name=self.extend_name('internal_network'),
            session=self.get_session(),
            input_shape=self.input_shape,
            layer_shapes=self.layer_shapes[:-1],
            activations=self.internal_activations,
            input_node=self.input_node
        )

        self.output_layer = VariationalLayer(
            name=self.extend_name('output_layer'),
            session=self.get_session(),
            input_shape=([self.input_shape] + self.layer_shapes)[-2],
            output_shape=self.layer_shapes[-1],
            means_activation=self.means_activation,
            stddevs_activation=self.stddevs_activation,
            input_node=self.internal_network.output_node
        )

        self.means_output_node = self.output_layer.means_output_node
        self.stddevs_output_node = self.output_layer.stddevs_output_node
        self.sample_node = self.output_layer.sample_node

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.internal_network.get_variables() + self.output_layer.get_variables()
