from wise.networks.layer import layer


class NoiseLayer(Layer):
    """
    Abstract base class for layers which serve only to add noise
    to their inputs.
    """

    def __init__(self, name, session, input_shape, noise_function,
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> (tf.Tensor -> (tf.Tensor, [tf.Variable]))
            -> tf.Tensor? -> tf.String? -> NoiseLayer
        Create a noise layer which adds independent noise to each of the elements
        of the input tensor, based on a function which returns a tensorflow tensor
        as output and a list of any variables used for the noise generation.
        """
        super().__init__(name, session, input_shape, output_shape=input_shape,
            input_node=input_node, save_location=save_location)
        
        self.output_node, self.variables = noise_function(self.input_node)

    def get_variables(self):
        """
        () -> ()
        """
        return self.variables
