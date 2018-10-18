from wise.networks.layer import Layer
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
