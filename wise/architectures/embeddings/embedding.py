from wise.networks.network import Network
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.training.samplers.dataset import DataSetSampler
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.resampled import BinomialResampler
from wise.training.samplers.feeddict import FeedDictSampler
from wise.util.tensors import int_placeholder_node, \
    placeholder_node, glorot_initialised_vars
from random import randint
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
            -> (String -> tf.Session -> Int -> tf.Node 
                -> (Network, tf.Node, Int))
            -> (a -> [Float]) -> (tf.Tensor -> String -> tf.Tensor)
            -> String? -> Embedding
        Create an embedding for the given list of items.  The discriminator
        generator should be a function which, given a name, session, 
        input dimension, and input node, returns a network as well
        as the node from the network which should be used as its
        output and the dimensionality of that output node (which must
        be a vector).
        """
        super().__init__(name, session, save_location)

        self.items = items
        self.n_items = len(items)
        self.embedding_dimension = embedding_dimension
        self.discriminator_builder = discriminator_builder
        self.truth_function = truth_function
        self.activation = activation

        self.embeddings = None
        self.indices_input = None
        self.lookups = None
        self.activated_lookups = None

        self.discriminator = None
        self.output_node = None
        self.output_dimension = None

        self.index_lookup = None

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
        self.activated_lookups = self.activation(self.lookups,
            self.extend_name('activated_embeddings'))

        self.discriminator, self.output_node, self.output_dimension = \
            self.discriminator_builder(self.extend_name('discriminator'),
                self.get_session(), self.embedding_dimension,
                self.activated_lookups)

        self.index_lookup = self._invert_items(self.items)
        
    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return [self.embeddings] + self.discriminator.get_variables()

    def sampler(self, target_node=None,
            resample=True, lazy=False, feed_dict=True):
        """
        Object? -> Bool? -> Bool? -> Bool? -> Sampler (Int, [Float])
        Return a sampler for training the embeddings and discriminator,
        optionally porting it to a feed dict sampler.
        """
        _target_node = target_node if target_node is not None else \
            self.target_node()

        if lazy:
            def single():
                n = randint(0, self.n_items - 1)
                return n, self.truth_function(self.items[n])
            sampler = AnonymousSampler(single=single)
        else:
            sampler = DataSetSampler(list(zip(range(self.n_items),
                [self.truth_function(i) for i in self.items])))

        if resample:
            sampler = BinomialResampler.halves_on_last_element_head(sampler)

        if not feed_dict:
            return sampler

        return FeedDictSampler(sampler, {
            self.indices_input: lambda t: t[0],
            _target_node: lambda t: t[1]
        })

    def target_node(self):
        """
        () -> tf.Placeholder
        Return a target node suitable for training the embeddings.
        """
        return placeholder_node(self.extend_name('target'),
            [self.output_dimension], dynamic_dimensions=1)

    @staticmethod
    def feedforward_classification_discriminator(hidden_layer_shapes):
        """
        [Int] -> (String -> tf.Session -> Int
            -> tf.Node -> (Network, tf.Node, Int))
        Create a function that can be called to return a standard
        feedforward network which has ReLU activations on all
        hidden layers, and a sigmoid activation on its single
        output node.
        """
        def build(name, session, input_dimension, input_node):
            net = FeedforwardNetwork(name, session, [input_dimension],
                hidden_layer_shapes + [[1]], activations=
                Activation.all_except_last(Activation.LEAKY_RELU,
                Activation.SIGMOID), input_node=input_node)
            return net, net.output_node, 1
        return build

    def _invert_items(self, items):
        """
        [a] -> Dict Int a
        Produce a lookup dictionary, retrieving an item's index
        given the item.
        """
        return { o: i for i, o in zip(range(self.n_items), items)}

    def item_by_index(self, n):
        """
        Int -> a
        """
        return self.items[n]

    def items_by_index(self, ns):
        """
        [Int] -> [a]
        """
        return [self.items[n] for n in ns]

    def index_by_item(self, item):
        """
        a -> Int
        """
        return self.index_lookup[item]

    def indices_by_idem(self, items):
        """
        [a] -> [Int]
        """
        return [self.index_lookup[item] for item in items]

    def embedding_by_index(self, n):
        """
        Int -> [Float]
        """
        return self.feed(self.activated_lookups,
            {self.indices_input: [n]})[0]

    def embeddings_by_index(self, ns):
        """
        [Int] -> [[Float]]
        """
        return self.feed(self.activated_lookups, {self.indices_input: ns})

    def embedding_by_item(self, item):
        """
        a -> [Float]
        """
        return self.embedding_by_index(self.index_by_item(item))

    def embeddings_by_item(self, items):
        """
        [a] -> [[Float]]
        """
        return self.embeddings_by_index(self.indices_by_idem(items))


class VariationalEmbedding(Network):
    """
    Describes an embedding whose values are sampled from a
    Gaussian parameterised by a tensor of means and
    standard deviations.
    """
    
    def __init__(self, name, session, items, embedding_dimension,
            discriminator_builder, truth_function,
            activation=Activation.IDENTITY, save_location=None):
        """
        String -> tf.Session -> [object] -> Int
            -> (String -> tf.Session -> Int -> tf.Node 
                -> (Network, tf.Node, Int))
            -> (a -> [Float]) -> (tf.Tensor -> String -> tf.Tensor)
            -> String? -> Embedding
        Create an embedding for the given list of items.  The discriminator
        generator should be a function which, given a name, session, 
        input dimension, and input node, returns a network as well
        as the node from the network which should be used as its
        output and the dimensionality of that output node (which must
        be a vector).
        """
        super().__init__(name, session, save_location)

        self.items = items
        self.n_items = len(items)
        self.embedding_dimension = embedding_dimension
        self.discriminator_builder = discriminator_builder
        self.truth_function = truth_function
        self.activation = activation

        self.embedding_means = None
        self.embedding_stddevs = None
        self.indices_input = None
        self.mean_lookups = None
        self.stddev_lookups = None
        self.variational_lookups = None
        self.activated_means = None
        self.activated_variational_lookups = None

        self.discriminator = None
        self.variational_discriminator = None
        self.output_node = None
        self.variational_output_node = None

        self.output_dimension = None
        self.index_lookup = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.embedding_means = glorot_initialised_vars(
            self.extend_name('embedding_means'),
            [self.n_items, self.embedding_dimension])
        self.embedding_stddevs = glorot_initialised_vars(
            self.extend_name('embedding_stddevs'),
            [self.n_items, self.embedding_dimension])

        self.indices_input = int_placeholder_node(self.extend_name(
            'indices_input'), [], 1)

        self.mean_lookups = tf.nn.embedding_lookup(self.embedding_means,
            self.indices_input, name=self.extend_name('mean_lookups'))
        self.stddev_lookups = tf.nn.embedding_lookup(self.embedding_stddevs,
            self.indices_input, name=self.extend_name('stddev_lookups'))

        self.variational_lookups = tf.add(self.mean_lookups,
            tf.multiply(tf.random_normal([self.embedding_dimension]),
            self.stddev_lookups), name=self.extend_name('variational_lookups'))
        
        self.activated_means = self.activation(self.mean_lookups,
            self.extend_name('activated_means'))

        self.activated_variational_lookups = self.activation(
            self.variational_lookups, self.extend_name(
                'activated_variational_lookups'))

        self.discriminator, self.output_node, self.output_dimension = \
            self.discriminator_builder(self.extend_name('discriminator'),
                self.get_session(), self.embedding_dimension,
                self.activated_means)

        self.variational_discriminator, self.variational_output_node, _ = \
            self.discriminator_builder(self.extend_name(
                'variational_discriminator'), self.get_session(),
                self.embedding_dimension, self.activated_variational_lookups)

        self.index_lookup = self._invert_items(self.items)

    def embedding_by_index(self, n, sampled=False):
        """
        Int -> Bool? -> [Float]
        """
        node = self.activated_variational_lookups if sampled \
            else self.activated_means
        return self.feed(node, {self.indices_input: [n]})[0]

    def embeddings_by_index(self, ns, sampled=False):
        """
        [Int] -> Bool? -> [[Float]]
        """
        node = self.activated_variational_lookups if sampled \
            else self.activated_means
        return self.feed(node, {self.indices_input: ns})
