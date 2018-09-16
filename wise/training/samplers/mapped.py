from wise.training.samplers.sampler import Sampler


class MappedSampler(Sampler):
    """
    A sampler whose output is a mapped version of the output of
    another sampler.
    """

    def __init__(self, sampler, f):
        """
        Sampler a -> (a -> b) -> Sampler b
        Create a sampler whose output is the result of another sampler's
        output, mapped according to some function.
        """
        self.sampler = sampler
        self.transfer = f

    def single(self):
        """
        () -> a
        """
        return self.transfer(self.sampler.single())

    def batch(self, n):
        """
        Int -> [a]
        """
        return [self.transfer(sample) for sample in self.sampler.batch(n)]
