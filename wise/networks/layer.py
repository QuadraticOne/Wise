from wise.networks.network import Network
from wise.util.tensors import placeholder_node


class Layer(Network):
    """
    Defines the behaviour of a single network layer.
    """

    def __init__(self, name, session, input_shape, output_shape,
            input_node=None, save_location=None):
        """
        String -> tf.Session -> [Int] -> [Int]
            -> tf.Tensor? -> String? -> Layer
        """
        super().__init__(name, session, save_location)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_node = input_node
        self.output_node = None

        if self.input_node is None:
            self.input_node = placeholder_node(self.extend_name('input_node'),
                self.input_shape, 1)

    def get_input_node(self):
        """
        () -> tf.Tensor
        """
        return self.input_node

    def get_output_node(self):
        """
        () -> tf.Tensor
        """
        return self.output_node

    def get_variables():
        """
        () -> [tf.Variable]
        """
        raise NotImplementedError()
