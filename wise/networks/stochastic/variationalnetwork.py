from wise.networks.mlp import MLP
from wise.networks.stochastic.variationallayer import VariationalLayer
from wise.networks.deterministic.feedforwardlayer import FeedforwardLayer
from wise.networks.activation import Activation

class VariationalNetwork(MLP):
    """
    Similar to a standard feedforward network but with a variational output
    layer.
    """

    def __init__(self, name, session, input_shape, layer_shapes,
            internal_activations=Activation.all(Activation.DEFAULT),
            means_activation=Activation.TANH, stddevs_activation=Activation.SIGMOID,
            input_node=None, save_location=None, batch_normalisation=False):
        """
        String -> tf.Session -> [Int] -> [[Int]] -> (Int -> [(tf.Tensor -> String -> tf.Tensor)])?
            -> (tf.Tensor -> String -> tf.Tensor)? -> (tf.Tensor -> String -> tf.Tensor)?
            -> tf.Tensor? -> String? -> Bool? -> VariationalNetwork
        """
        self.internal_activations = internal_activations
        self.means_activation = means_activation
        self.stddevs_activation = stddevs_activation

        self.batch_normalisation = batch_normalisation

        super().__init__(name, session, input_shape, self.make_constructors_generator,
            layer_shapes, input_node, save_location)

        self.means_output_node = self.layers[-1].means_output_node
        self.stddevs_output_node = self.layers[-1].stddevs_output_node

    def make_constructors_generator(self, n_layers):
        """
        Int -> [String -> tf.Session -> [Int] -> [Int]
            -> tf.Tensor? -> String? -> Layer]
        Given the parameters passed to the network, when given the number
        of layers the network will have, create a list of functions that
        produce each layer given the layer size.
        """
        def make_deterministic_constructor(activation):
            def deterministic_constructor(name, session, input_shape,
                    output_shape, input_node=None, save_location=None):
                return FeedforwardLayer(
                    name=name,
                    session=session,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    activation=activation,
                    input_node=input_node,
                    save_location=save_location,
                    batch_normalisation=self.batch_normalisation
                )
            return deterministic_constructor

        def variational_constructor(name, session, input_shape,
                output_shape, input_node=None, save_location=None):
            return VariationalLayer(
                name=name,
                session=session,
                input_shape=input_shape,
                output_shape=output_shape,
                means_activation=self.means_activation,
                stddevs_activation=self.stddevs_activation,
                input_node=input_node,
                save_location=save_location,
                batch_normalisation=self.batch_normalisation
            )

        return [make_deterministic_constructor(activation) \
            for activation in self.internal_activations(n_layers - 1)] + \
            [variational_constructor]
