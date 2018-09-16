from wise.networks.network import Network
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation


class Autoencoder(Network):
    """
    An autoencoding neural network, useful for dimensionality reduction
    and feature extraction.
    """

    def __init__(self, name, session, input_shape, encoder_layer_shapes,
            decoder_layer_shapes=None, encoder_activations=Activation.all(Activation.DEFAULT),
            decoder_activations=Activation.all(Activation.DEFAULT),
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [[Int]] -> [[Int]]? ->
            (Int -> [(tf.Tensor -> String -> tf.Tensor)])? ->
            (Int -> [(tf.Tensor -> String -> tf.Tensor)])? -> tf.Tensor? -> String? -> Autoencoder
        Create an autoencoder from the given parameters.  To have the decoder mirror the
        encoder, simply leave the `decoder_layer_shapes` parameter as its default.
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.encoder_layer_shapes = encoder_layer_shapes
        self.decoder_layer_shapes = decoder_layer_shapes if decoder_layer_shapes is not None \
            else self._mirror_encoder_layers()
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations

        self.input_node = input_node
        self.encoder = None
        self.latent_node = None
        self.decoder = None
        self.reconstruction_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.encoder = FeedforwardNetwork(
            name=self.extend_name('encoder'),
            session=self.get_session(),
            input_shape=self.input_shape,
            layer_shapes=self.encoder_layer_shapes,
            activations=self.encoder_activations,
            input_node=self.input_node
        )
        self.input_node = self.encoder.input_node
        self.latent_node = self.encoder.output_node
        self.decoder = FeedforwardNetwork(
            name=self.extend_name('decoder'),
            session=self.get_session(),
            input_shape=self.encoder_layer_shapes[-1],
            layer_shapes=self.decoder_layer_shapes,
            activations=self.decoder_activations,
            input_node=self.latent_node
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
