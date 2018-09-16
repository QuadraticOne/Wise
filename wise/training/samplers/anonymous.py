from wise.training.samplers.sampler import Sampler


class AnonymousSampler(Sampler):
    """
    A sampler whose single and batch functions are provided as inputs.
    Good for creating small, simple samplers quickly and succinctly
    from lambdas.
    """

    def __init__(self, single=None, batch=None):
        """
        (() -> a) -> (Int -> [a]) -> Sampler a
        """
        self._single = single
        self._batch = batch

        self._check_lambdas()

        self._single_call = self._single if self._single is not None \
            else lambda: self._batch(1)
        self._batch_call = self._batch if self._batch is not None \
            else lambda n: [self._single() for _ in range(n)]

    def _check_lambdas(self):
        """
        () -> ()
        Check that at least one of the two functions is defined.
        """
        if self._single is None and self._batch is None:
            raise ValueError('Either `single` or `batch` must not be None.')

    def single(self):
        """
        () -> a
        """
        return self._single_call()

    def batch(self, n):
        """
        Int -> [a]
        """
        return self._batch_call(n)
