class Sampler:
    """
    Base class from which samplers are derived, which abstract away
    the process of retrieving and processing input data.  If either
    `single` or `batch` is defined then the other will be derived,
    but at least one must be implemented.
    """
    
    def single(self):
        """
        () -> a
        Return a single sample from the sampler.
        """
        return self.batch(1)

    def batch(self, n):
        """
        Int -> [a]
        Return a batch of size n from the sampler.
        """
        return [self.single() for _ in range(n)]
