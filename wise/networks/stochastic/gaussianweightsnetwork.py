from wise.networks.mlp import MLP
from wise.networks.stochastic.gaussianweightslayer import GaussianWeightsLayer
from wise.networks.activation import Activation


class GaussianWeightsNetwork(MLP):
    """
    Feedforward neural netowrk, consisting of a number of layers
    of fully connected neurons whose weights are sampled from distinct
    Gaussian distributions.
    """

    def __init__(self, name, session, input_shape, layer_shapes, activations=
            Activation.all(Activation.DEFAULT), input_node=None,
            save_location=None, batch_normalisation=False):
        """
        String -> tf.Session -> [Int] -> [[Int]] ->
            (Int -> [(tf.Tensor -> String -> tf.Tensor)]) -> tf.Tensor? -> String? ->
            Bool? -> FeedforwardNetwork
        Create a network with Gaussian sampled weights from the given parameters.
        """
        self.batch_normalisation = batch_normalisation
        super().__init__(
            name=name,
            session=session,
            input_shape=input_shape,
            layer_constructors_generator=lambda n: [self.make_layer_constructor(activation) \
                for activation in activations(n)],
            layer_shapes=layer_shapes,
            input_node=input_node,
            save_location=save_location
        )

    def make_layer_constructor(self, activation):
        def constructor(name, session, input_shape,
                output_shape, input_node, save_location=None):
            """
            String -> tf.Session -> [Int] -> [Int]
                -> tf.Tensor? -> String? -> FeedforwardLayer
            """
            return GaussianWeightsLayer(
                name=name,
                session=session,
                input_shape=input_shape,
                output_shape=output_shape,
                activation=activation,
                input_node=input_node,
                save_location=save_location,
                batch_normalisation=self.batch_normalisation
            )
        return constructor
