from wise.networks.layer import Layer
from wise.util.tensors import glorot_initialised_vars
import tensorflow as tf


class NoiseLayer(Layer):
    """
    Abstract base class for layers which serve only to add noise
    to their inputs.
    """

    def __init__(self, name, session, input_shape, input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> tf.Tensor? -> tf.String? -> NoiseLayer
        Create a noise layer which adds independent noise to each of the elements
        of the input tensor, based on a function which returns a tensorflow tensor
        as output and a list of any variables used for the noise generation.
        """
        super().__init__(name, session, input_shape, output_shape=input_shape,
            input_node=input_node, save_location=save_location)
        
        self.output_node, self.variables = self.noise_function(self.input_node)

    def noise_function(self, input_tensor):
        """
        tf.Tensor -> (tf.Tensor, [tf.Variable])
        Given an input tensor, return an output tensor - which is the input tensor
        with noise added to each element - and any variables required to do so.
        """
        raise NotImplementedError()

    def get_variables(self):
        """
        () -> ()
        """
        return self.variables


class GaussianNoiseLayer(NoiseLayer):
    """
    Adds IDD Gaussian noise to the elements of a tensor with a
    mean of zero and a standard deviation that is constant across
    all elements.
    """

    def __init__(self, name, session, input_shape, noise_stddev,
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> Float -> tf.Tensor? -> String?
            -> GaussianNoiseLayer
        """
        self.noise_stddev = noise_stddev
        super().__init__(name, session, input_shape, input_node, save_location)

    def noise_function(self, input_node):
        """
        tf.Tensor -> (tf.Tensor, [tf.Variable])
        """
        return tf.add(self.get_input_node(),
            tf.random_normal(self.input_shape, stddev=self.noise_stddev,
                name=self.extend_name('gaussian_noise')),
            name=self.extend_name('output_node')), []


class ElementwiseGaussianNoiseLayer(NoiseLayer):
    """
    Adds IDD Gaussian noise to the elements of a tensor, where the
    standard deviation of the generated noise is given by an independent
    variable tensor, and the mean of the Gaussian noise is zero.
    """

    def __init__(self, name, session, input_shape, input_node=None,
            save_location=None):
        """
        String -> tf.Session -> [Int] -> tf.Tensor? -> String?
            -> ElementwiseGaussianNoiseLayer
        """
        super().__init__(name, session, input_shape, input_node, save_location)

    def noise_function(self, input_node):
        """
        tf.Tensor -> (tf.Tensor, [tf.Variable])
        """
        stddevs = glorot_initialised_vars(self.extend_name('standard_deviations'),
            self.input_shape)
        noise = tf.random_normal(self.input_shape, name=self.extend_name('noise'))
        return tf.add(input_node, tf.multiply(stddevs, noise),
            name=self.extend_name('output_node')), [stddevs]


class StochasticBinaryUnitLayer(NoiseLayer):
    """
    A layer which randomly turns off (sets to zero) each input
    with a given probability.
    """

    def __init__(self, name, session, input_shape, input_node=None,
            off_probability=0.5, save_location=None):
        """
        String -> tf.Session -> [Int] -> tf.Tensor? -> Float? -> String?
            -> StochasticBinaryUnitLayer
        """
        self.off_probability = off_probability
        super().__init__(name, session, input_shape, input_node, save_location)

    def noise_function(self, input_node):
        """
        tf.Tensor -> (tf.Tensor, [tf.Variable])
        """
        uniform = tf.random_uniform(self.input_shape, 0, 1,
            name=self.extend_name('random_uniform'))
        is_off = tf.less(self.off_probability, uniform,
            name=self.extend_name('is_off'))
        return tf.multiply(input_node, tf.to_float(is_off),
            name=self.extend_name('output_node')), []
