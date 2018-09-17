from wise.networks.network import Network
from wise.networks.activation import Activation
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.stochastic.variationalnetwork import VariationalNetwork
from wise.util.tensors import placeholder_node


class VAE(Network):
    """
    A variational autoencoder - similar to a regular autoencoder, but learns
    a probability distribution in the latent space.  Can be used for sampling
    from the latent space or for dimensionality reduction.
    """

    def __init__(self, name, session, input_shape, encoder_layer_shapes,
            decoder_layer_shapes=None, encoder_activations=Activation.all(Activation.DEFAULT),
            encoder_means_activation=Activation.TANH, encoder_stddevs_activation=Activation.SIGMOID,
            decoder_activations=Activation.all(Activation.DEFAULT),
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [[Int]] -> [[Int]]? -> (Int ->
            [(tf.Tensor -> String -> tf.Tensor)])? -> (tf.Tensor -> String -> tf.Tensor)?
            -> (tf.Tensor -> String -> tf.Tensor)? -> (Int -> [(tf.Tensor -> String -> tf.Tensor)])?
            -> tf.Tensor? -> String?
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.encoder_layer_shapes = encoder_layer_shapes
        self.decoder_layer_shapes = decoder_layer_shapes if decoder_layer_shapes is not None \
            else self._mirror_encoder_layers()
        self.encoder_activations = encoder_activations
        self.encoder_means_activation = encoder_means_activation
        self.encoder_stddevs_activation = encoder_stddevs_activation
        self.decoder_activations = decoder_activations

        self.input_node = input_node
        self.encoder = None
        self.means_output_node = None
        self.stddevs_output_node = None
        self.sample_node = None
        self.decoder = None
        self.reconstruction_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        if self.input_node is None:
            self.input_node = placeholder_node(self.extend_name('input_node'),
                self.input_shape, dynamic_dimensions=1)

        self.encoder = VariationalNetwork(
            name=self.extend_name('encoder'),
            session=self.get_session(),
            input_shape=self.input_shape,
            layer_shapes=self.encoder_layer_shapes,
            internal_activations=self.encoder_activations,
            means_activation=self.encoder_means_activation,
            stddevs_activation=self.encoder_stddevs_activation,
            input_node=self.input_node
        )
        self.means_output_node = self.encoder.means_output_node
        self.stddevs_output_node = self.encoder.stddevs_output_node
        self.sample_node = self.encoder.sample_node

        self.decoder = FeedforwardNetwork(
            name=self.extend_name('decoder'),
            session=self.get_session(),
            input_shape=self.encoder_layer_shapes[-1],
            layer_shapes=self.decoder_layer_shapes,
            activations=self.decoder_activations,
            input_node=self.sample_node
        )
        self.reconstruction_node = self.decoder.output_node

    def _mirror_encoder_layers(self):
        """
        () -> [[Int]]
        Copy the layers of the encoder and mirror them, producing the default layers
        of the decoder.
        """
        return self.encoder_layer_shapes[::-1][1:] + [self.input_shape]

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.encoder.get_variables() + self.decoder.get_variables()
