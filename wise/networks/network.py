from wise.util.io import IO
import tensorflow as tf


class Network:
    """
    Abstract class for Wise networks.
    """
    
    def __init__(self, name, session, save_location=None):
        self.name = name
        self.session = session
        self.save_location = save_location

    def get_name(self):
        """
        () -> Name
        """
        return self.name

    def get_session(self):
        """
        () -> tf.Session
        """
        return self.session()

    def extend_name(self, extension):
        """
        String -> String
        """
        return self.name + '.' + extension

    def get_variables(self):
        """
        () -> [tf.Variable]
        Return all the variables upon which this network depends.
        """
        raise NotImplementedError()

    def feed(self, outputs, feed_dict):
        """
        [Tensor] -> Dict Tensor Array -> [Array]
        """
        return self.session.feed(outputs, feed_dict=feed_dict)

    def save(self):
        """
        () -> ()
        Save the values of the network's variables to its save location.
        """
        self._check_save_location()
        io = IO(self.save_location)
        io.save_session(self.session, 'parameters', variables=self.get_variables())

    def restore(self):
        """
        () -> ()
        Restore the values of the network's variables from its save location.
        """
        self._check_save_location()
        io = IO(self.save_location)
        io.restore_session(self.session, 'parameters', variables=self.get_variables())

    def initialise_variables(self):
        """
        () -> ()
        Initialise the network's variables.
        """
        for variable in self.get_variables():
            self.feed(tf.variables_initializer())

    def initialise_and_restore(self):
        """
        () -> ()
        Initialise the network's variables then restore them.
        """
        self.initialise_variables()
        self.restore()
                
    def _check_save_location(self):
        """
        () -> ()
        """
        if self.save_location is None:
            raise ValueError('The network\'s save location is None.')

    def count_parameters(self):
        """
        () -> Int
        """
        return sum(self.feed([tf.size(v) for v in self.get_variables()], {}))
