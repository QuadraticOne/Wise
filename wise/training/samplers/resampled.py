from wise.training.samplers.sampler import Sampler
from queue import Queue
from random import uniform


class BinomialResampler(Sampler):
    """
    Takes an input sampler and returns an output sampler wherein
    the outputs are resampled to obey some binomial distribution 
    according to the result of a predicate.
    """

    def __init__(self, sampler, predicate, p_true, max_queue_size=256):
        """
        Sampler a -> (a -> Bool) -> Float -> Int? -> BinomialResampler a
        Define a binomially resampled version of another sampler based
        on a predicate and the desired probability that an output from the
        new sampler satisfies the predicate.
        """
        self.sampler = sampler
        self.predicate = predicate
        self.p_true = p_true

        self.max_queue_size = max_queue_size

        self.trues = Queue()
        self.falses = Queue()

    def _populate_to(self, n):
        """
        Int -> ()
        Populate both queues until they each have a size of at least n.
        """
        if self.max_queue_size < 2 * n:
            self.max_queue_size = 2 * n
        while self.trues.qsize() < n or self.falses.qsize() < n:
            new_trues, new_falses = _partition(self.sampler.batch(
                self.max_queue_size // 4), self.predicate)
            self._add_trues(new_trues)
            self._add_falses(new_falses)
        
    def _add_trues(self, new_trues):
        """
        [a] -> ()
        Add the given elements, which are assumed to satisfy the predicate,
        onto the end of the trues queue while its length is below the
        specified maximum.
        """
        for t in new_trues:
            if self.trues.qsize() < self.max_queue_size:
                self.trues.put(t)
            else:
                break

    def _add_falses(self, new_falses):
        """
        [a] -> ()
        Add the given elements, which are assumed to not satisfy the predicate,
        onto the end of the falses queue while its length is below the
        specified maximum.
        """
        for f in new_falses:
            if self.falses.qsize() < self.max_queue_size:
                self.falses.put(f)
            else:
                break

    def single(self):
        """
        () -> a
        """
        self._populate_to(1)
        q = uniform(0, 1)
        if q < self.p_true:
            return self.trues.get()
        else:
            return self.falses.get()

    def batch(self, n):
        """
        Int -> [a]
        """
        self._populate_to(n)
        output = []
        for _ in range(n):
            q = uniform(0, 1)
            if q < self.p_true:
                output.append(self.trues.get())
            else:
                output.append(self.falses.get())
        return output

    @staticmethod
    def halves_on_last_element(sampler, p_true=0.5, threshold=0.5,
            max_queue_size=512):
        """
        Sampler a -> Float? -> Float? -> Int? -> BinomialResampler a
        Resample the given sampler such that the probability of the last
        element of its output being greater than the threshold is the
        given probability.
        """
        return BinomialResampler(sampler, lambda t: t[-1] > threshold, p_true,
            max_queue_size=max_queue_size)

    @staticmethod
    def halves_on_last_element_head(sampler, p_true=0.5, threshold=0.5,
            max_queue_size=512):
        """
        Sampler a -> Float? -> Float? -> Int? -> BinomialResampler a
        Resample the given sampler such that the probability of its last
        element's head being greater than the threshold is the
        given probability.
        """
        return BinomialResampler(sampler, lambda t: t[-1][0] > threshold, p_true,
            max_queue_size=max_queue_size)


def _partition(ls, f):
    """
    [a] -> (a -> Bool) -> ([a], [a])
    Split a list into two lists of those elements which satisfy the
    predicate and those element that do not.
    """
    trues = []
    falses = []
    for el in ls:
        if f(el):
            trues.append(el)
        else:
            falses.append(el)
    return trues, falses
