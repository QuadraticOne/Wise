from wise.networks.network import Network
from wise.networks.activation import Activation
from wise.util.tensors import int_placeholder_node, glorot_initialised_vars
import tensorflow as tf


class Embedding(Network):
    """
    A class which allows the training of embeddings for objects
    which would not otherwise be representable by a vector.
    """

    def __init__(self, name, session, items, embedding_dimension,
            discriminator_builder, truth_function,
            activation=Activation.IDENTITY, save_location=None):
        """
        String -> tf.Session -> [object] -> Int
            -> (String -> tf.Session -> Int -> tf.Node -> (Network, tf.Node))
            -> (a -> Bool) -> (tf.Tensor -> String -> tf.Tensor)
            -> String? -> Embedding
        Create an embedding for the given list of items.  The discriminator
        generator should be a function which, given a name, session, 
        input dimension, and input node, returns a network as well
        as the node from the network which should be used as its
        output.
        """
        super().__init__(name, session, save_location)

        self.items = items
        self.n_items = len(items)
        self.embedding_dimension = embedding_dimension
        self.discriminator_builder = discriminator_builder
        self.activation = activation

        self.embeddings = None
        self.indices_input = None
        self.lookups = None

        self.discriminator = None
        self.output_node = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.embeddings = glorot_initialised_vars(
            self.extend_name('embeddings'),
            [self.n_items, self.embedding_dimension])
        self.indices_input = int_placeholder_node(self.extend_name(
            'indices_input'), [], 1)
        self.lookups = tf.nn.embedding_lookup(self.embeddings,
            self.indices_input, name=self.extend_name('lookups'))
        
    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.embeddings + self.discriminator.get_variables()
