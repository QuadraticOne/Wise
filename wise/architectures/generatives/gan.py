from wise.networks.network import Network


class GAN(Network):
    """
    A generative adversarial network that learns to produce samples
    from a distribution by pitting a generator and discriminator
    against each other in a minimax game.
    """

    def __init__(self, name, session, save_location=None):
        """
        String -> tf.Session -> String? -> GAN
        """
        pass

    def _initialise(self):
        """
        () -> ()
        """
        pass

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        pass
