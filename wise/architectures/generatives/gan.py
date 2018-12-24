from wise.networks.network import Network
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.feeddict import FeedDictSampler
from wise.util.tensors import placeholder_node
from wise.util.training import default_adam_optimiser
from random import choice
import tensorflow as tf


class GAN(Network):
    """
    A generative adversarial network that learns to produce samples
    from a distribution by pitting a generator and discriminator
    against each other in a minimax game.
    """

    def __init__(self, name, session, noise_dimension, latent_dimension,
            generator_builder, discriminator_builder, save_location=None):
        """
        String -> tf.Session -> Int -> Int
            -> (String -> tf.Session -> Int -> tf.Tensor
                -> Int -> (Network, tf.Tensor)) or [Int]
            -> (String -> tf.Session -> Int -> tf.Tensor
                -> Int -> (Network, tf.Tensor)) or [Int]
            -> String? -> GAN
        """
        super().__init__(name, session, save_location)

        self.noise_dimension = noise_dimension
        self.latent_dimension = latent_dimension

        self.generator_builder = generator_builder
        self.discriminator_builder = discriminator_builder

        self.noise_vector = None
        self.real_input = None
        self.fake_input = None

        # The switch is True for real inputs and False for fake inputs
        self.switch = None
        self.discriminator_input = None
        self.discriminator_output = None

        self.generator = None
        self.discriminator = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.noise_vector = tf.random_normal([self.noise_dimension],
            name=self.extend_name('noise'))
        self.real_input = placeholder_node(self.extend_name('real_input'),
            [self.latent_dimension], dynamic_dimensions=1)

        if type(self.generator_builder) == type([]):
            self.generator_builder = GAN.default_network(self.generator_builder)

        self.generator, self.fake_input = self.generator_builder(
            self.extend_name('generator'), self.get_session(),
            self.noise_dimension, self.noise_vector, self.latent_dimension    
        )
        
        self.switch = tf.placeholder(tf.bool, shape=[None],
            name=self.extend_name('switch'))
        self.discriminator_input = tf.where(self.switch,
            self.real_input, self.fake_input)
        
        if type(self.discriminator_builder) == type([]):
            self.discriminator_builder = GAN.default_network(
                self.discriminator_builder)

        self.discriminator, self.discriminator_output = \
            self.discriminator_builder(
                self.extend_name('discriminator'), self.get_session(),
                self.latent_dimension, self.discriminator_input, 1            
            )

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.generator.get_variables() + \
            self.discriminator.get_variables()

    def loss_nodes(self):
        """
        () -> (tf.Tensor, tf.Tensor)
        Return loss nodes for the generator and discriminator
        respectively.  Note that the generator loss node is
        only valid when the discriminator is classifying a
        fake input.
        """
        generator_loss = tf.losses.log_loss(
            0., self.discriminator_output)
        discriminator_loss = tf.losses.log_loss(
            1., self.discriminator_output)
        return generator_loss, discriminator_loss

    def samplers(self, examples, as_feed_dict=True):
        """
        [[Float]] -> Bool?
            -> (Sampler (Bool, [Float]), Sampler (Bool, [Float]))
        Return two samplers; one for training on real inputs,
        and one for training on false inputs.
        """
        real_sampler = AnonymousSampler(
            single=lambda: (True, choice(examples)))
        fake_sampler = AnonymousSampler(
            single=lambda: (False, choice(examples)))
        
        if as_feed_dict:
            def to_feed_dict(s):
                return FeedDictSampler(s, {
                    self.switch: lambda t: t[0],
                    self.real_input: lambda t: t[1]
                })
            real_sampler = to_feed_dict(real_sampler)
            fake_sampler = to_feed_dict(fake_sampler)
        
        return real_sampler, fake_sampler

    @staticmethod
    def default_network(hidden_layer_shapes):
        """
        [Int] -> (String -> tf.Session -> Int -> tf.Tensor -> Int
            -> (Network, tf.Tensor))
        Return a function that builds a default feedforward network
        to be used either as a generator or discriminator.
        """
        def build(name, session, input_dimension, input_node,
                output_dimension):
            net = FeedforwardNetwork(
                name=name,
                session=session,
                input_shape=[input_dimension],
                layer_shapes=hidden_layer_shapes + [[output_dimension]],
                activations=Activation.all_except_last(
                    Activation.LEAKY_RELU, Activation.SIGMOID),
                input_node=input_node
            )
            return net, net.output_node
        return build
