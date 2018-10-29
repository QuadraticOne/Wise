from wise.networks.mlp import MLP
from wise.networks.deterministic.feedforwardlayer import FeedforwardLayer
from wise.networks.activation import Activation


class FeedforwardNetwork(MLP):
    """
    Feedforward neural netowrk, consisting of a number of layers
    of fully connected neurons.
    """

    def __init__(self, name, session, input_shape, layer_shapes, activations=
            Activation.all(Activation.DEFAULT), input_node=None, save_location=None,
            batch_normalisation=False):
        """
        String -> tf.Session -> [Int] -> [[Int]] ->
            (Int -> [(tf.Tensor -> String -> tf.Tensor)]) -> tf.Tensor? -> String? ->
            Bool? -> FeedforwardNetwork
        Create a feedforward network from the given parameters.
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
        return constructor
