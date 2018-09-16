class FeedDictSampler:
    """
    A class similar to Sampler which wraps the results of a sampler
    in a feed_dict for feeding into a network.
    """

    def __init__(self, sampler, key_lambdas):
        """
        Sampler a -> Dict b (a -> c) -> FeedDictSampler c
        Take a sampler, as well as a dictionary of keys mapped to
        functions which extract a value from the output of that
        sampler, and return a FeedDictExtractor which wraps the
        results of that sampler in a feed_dict.
        """
        self.sampler = sampler
        self.key_lambdas = key_lambdas

    def single(self):
        """
        () -> Dict b [c]
        """
        return self.batch(1)

    def batch(self, n):
        """
        Int -> Dict b [c]
        """
        _batch = self.sampler.batch(n)
        return {
            k: [f(sample) for sample in _batch] \
                for k, f in self.key_lambdas.items()   
        }

    @staticmethod
    def from_indices(sampler, keys):
        """
        Sampler a -> [Tensor] -> FeedDictSampler b
        Take a sampler, which outputs a list or tuple of values, and return a
        FeedDictSampler which maps the ith element of the sampled tuples to the
        ith tensor in the provided list.
        """
        return FeedDictSampler(sampler, {
            # TODO: find a way to "save" the integer indices properly
            index_key[1]: lambda t: t[index_key[0]] for index_key in enumerate(keys)
        })
