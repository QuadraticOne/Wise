from wise.networks.network import Network
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.feeddict import FeedDictSampler
from wise.util.tensors import placeholder_node
from wise.util.training import default_adam_optimiser
from random import choice
import tensorflow as tf
import numpy as np


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
                -> Int -> (Network, tf.Tensor)) or [[Int]]
            -> (String -> tf.Session -> Int -> tf.Tensor
                -> Int -> (Network, tf.Tensor)) or [[Int]]
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
        self.input_is_real = None
        self.discriminator_input = None
        self.discriminator_output = None

        self.generator = None
        self.discriminator = None

        self.generator_loss = None
        self.discriminator_loss = None

        self.generator_optimiser = None
        self.discriminator_optimiser = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.noise_vector = placeholder_node(self.extend_name('noise'),
            [self.noise_dimension], dynamic_dimensions=1)
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
        self.input_is_real = tf.cast(self.switch, tf.float32)
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

    def set_loss_nodes(self):
        """
        () -> ()
        Set loss nodes for the generator and discriminator
        respectively.  Note that the generator loss node is
        only valid when the discriminator is classifying a
        fake input.
        """
        self.generator_loss = tf.losses.log_loss(
            1., self.discriminator_output)
        self.discriminator_loss = tf.losses.log_loss(
            self.input_is_real, self.discriminator_output)

    def samplers(self, examples, as_feed_dict=True):
        """
        [[Float]] -> Bool? -> (Sampler (Bool, [Float], [Float]),
            Sampler (Bool, [Float], [Float]))
        Return two samplers; one for training on real inputs,
        and one for training on false inputs.
        """        
        # TODO: this could be sped up by just using zeros for noise
        #       since it is not used anyway
        real_sampler = AnonymousSampler(single=lambda:
            (True, self.get_noise_vector(), choice(examples)))
        fake_sampler = AnonymousSampler(single=lambda:
            (False, self.get_noise_vector(), choice(examples)))
        
        if as_feed_dict:
            def to_feed_dict(s):
                return FeedDictSampler(s, {
                    self.switch: lambda t: t[0],
                    self.noise_vector: lambda t: t[1],
                    self.real_input: lambda t: t[2]
                })
            real_sampler = to_feed_dict(real_sampler)
            fake_sampler = to_feed_dict(fake_sampler)
        
        return real_sampler, fake_sampler

    def set_optimisers(self):
        """
        () -> ()
        Using the generator and discriminator loss nodes saved
        in the GAN object, create an optimiser for each network
        and save it in the GAN object.
        """
        self.generator_optimiser = default_adam_optimiser(
            self.generator_loss, self.extend_name('generator_optimiser'),
            self.generator.get_variables())
        self.discriminator_optimiser = default_adam_optimiser(
            self.discriminator_loss,
            self.extend_name('discriminator_optimiser'),
            self.discriminator.get_variables())

    def train(self, examples, epochs, steps_per_epoch, batch_size,
            evaluation_sample_size=-1):
        """
        [[Float]] -> Int -> Int -> Int -> Int? -> ()
        Train the GAN by alternating between training the
        discriminator and then both networks on each step.
        Default loss nodes, samplers, and optimisers are used.
        Logging will be done if the evaluation sample size is
        set to a strictly positive number.
        """
        real_sampler, fake_sampler = self.samplers(examples)
        try:
            for epoch in range(epochs):
                if evaluation_sample_size > 0:
                    real_disc_loss = self.feed(self.discriminator_loss,
                        real_sampler.batch(evaluation_sample_size))
                    fake_disc_loss, gen_loss = self.feed(
                        [self.discriminator_loss, self.generator_loss],
                        fake_sampler.batch(evaluation_sample_size)
                    )
                    disc_loss = 0.5 * (real_disc_loss + fake_disc_loss)
                    print('Epoch: {}\tGenerator loss: {}\tDiscriminator Loss: {}'
                        .format(epoch, gen_loss, disc_loss))
                self._perform_epoch(real_sampler, fake_sampler,
                    steps_per_epoch, batch_size)
        except KeyboardInterrupt:
            print('Training stopped early.')

    def _perform_epoch(self, real_sampler, fake_sampler,
            steps_per_epoch, batch_size):
        """
        Sampler -> Sampler -> Int -> Int -> ()
        Perform an epoch of training with the given training
        parameters.
        """
        for _ in range(steps_per_epoch):
            self.feed(self.discriminator_optimiser,
                real_sampler.batch(batch_size))
            self.feed([self.discriminator_optimiser, self.generator_optimiser],
                fake_sampler.batch(batch_size))

    def get_noise_vector(self):
        """
        () -> [Float]
        Return a valid noise vector.
        """
        return np.random.uniform(-1., 1., size=[self.noise_dimension])

    def get_generator_sample(self, noise_vector=None):
        """
        [Float]? -> [Float]
        Return a single sample from the generator.  If no noise
        vector is specified, one will be drawn from a uniform
        distribution.
        """
        if noise_vector is None:
            noise_vector = self.get_noise_vector()
        return self.feed(self.fake_input,
            {self.noise_vector: [noise_vector]})[0]

    def get_generator_samples(self, noise_vectors=None):
        """
        [[Float]]? -> [[Float]]
        Return a number of samples from the generator.  If an integer
        value is passed instead of a list of noise vectors, the
        noise vectors will be drawn from a uniform distribution.
        """
        if noise_vectors is None:
            raise Exception('Either a set of noise vectors or '
             + 'a batch size must be specified.')
        elif type(noise_vectors) == type(0):
            noise_vectors = [self.get_noise_vector() 
                for _ in range(noise_vectors)]
        return self.feed(self.fake_input, {self.noise_vector: noise_vectors})

    def evaluate_discriminator(self, examples, real_inputs, batch_size):
        """
        [[Float]] -> Bool -> Int -> Float
        Evaluate the accuracy of the discriminator on a number
        of samples from either the real or fake samplers.
        """
        sampler = self.samplers(examples)[0 if real_inputs else 1]
        target = 1. if real_inputs else 0.
        outputs = self.feed(self.discriminator_output,
            sampler.batch(batch_size))
        
        count = 0
        for output in outputs:
            if (output[0] < 0.5) == (target < 0.5):
                count += 1
        return count / batch_size

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
