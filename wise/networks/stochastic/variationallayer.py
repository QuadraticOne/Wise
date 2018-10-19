from wise.networks.layer import Layer
from wise.networks.activation import Activation
from wise.networks.deterministic.feedforwardlayer import FeedforwardLayer
from wise.util.tensors import placeholder_node
import tensorflow as tf


class VariationalLayer(Layer):
    """
    A variational layer, whose output is a tensor of Gaussian distributions
    from which samples can be drawn.
    """

    def __init__(self, name, session, input_shape, output_shape,
            means_activation=Activation.TANH, stddevs_activation=Activation.SIGMOID,
            output_activation=Activation.DEFAULT, input_node=None,
            batch_normalisation=False, save_location=None):
        """
        String -> tf.Session -> [Int] -> [Int] -> (tf.Tensor -> String -> tf.Tensor)?
            -> (tf.Tensor -> String -> tf.Tensor)?-> (tf.Tensor -> String -> tf.Tensor)?
            -> tf.Tensor? -> Bool? -> String? -> VariationalLayer
        """
        super().__init__(name, session, input_shape,
            output_shape, input_node, save_location)

        self.means_activation = means_activation
        self.stddevs_activation = stddevs_activation
        self.output_activation = output_activation

        self.means_layer = None
        self.stddevs_layer = None
        self.means_output_node = None
        self.stddevs_output_node = None

        self.batch_normalisation = batch_normalisation

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.means_layer = FeedforwardLayer(
            name=self.extend_name('means_layer'),
            session=self.get_session(),
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            activation=self.means_activation,
            input_node=self.input_node,
            batch_normalisation=self.batch_normalisation
        )
        self.means_output_node = self.means_layer.output_node

        self.stddevs_layer = FeedforwardLayer(
            name=self.extend_name('stddevs_layer'),
            session=self.get_session(),
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            activation=self.stddevs_activation,
            input_node=self.input_node,
            batch_normalisation=self.batch_normalisation
        )
        self.stddevs_output_node = self.stddevs_layer.output_node

        self.output_node = self.output_activation(
            tf.add(tf.multiply(tf.random_normal(self.output_shape),
            self.stddevs_output_node), self.means_output_node,
            name=self.extend_name('output_node')),
            self.extend_name('activated_output')
        )

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.means_layer.get_variables() + self.stddevs_layer.get_variables()
