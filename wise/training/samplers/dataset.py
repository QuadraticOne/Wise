from wise.training.samplers.sampler import Sampler
from random import choice


class DataSetSampler(Sampler):
    """
    A sampler which draws data from a predefined list of points.
    """

    def __init__(self, points):
        """
        [a] -> DataSampler a
        """
        self.points = points
    
    def single(self):
        """
        () -> a
        """
        return choice(self.points)

    def batch(self, n):
        """
        Int -> [a]
        """
        return [choice(self.points) for _ in range(n)]

    @staticmethod
    def from_sampler(sampler, data_set_size):
        """
        Sampler a -> Int -> DataSetSampler a
        Create a concrete data set, containing the specified number
        of points, from another sampler.  Then return it as an
        independent sampler.
        """
        return DataSetSampler(sampler.batch(data_set_size))

    @staticmethod
    def training_validation_test(sampler, training_set_size,
            validation_set_size, test_set_size):
        """
        Sampler a -> Int -> Int -> Int 
            -> (DataSetSampler a, DataSetSampler a, DataSetSampler a)
        Create three samplers which sample from concrete and distinct
        data sets; one each for a training, validation, and test set.
        """
        return (
            DataSetSampler.from_sampler(sampler, training_set_size),
            DataSetSampler.from_sampler(sampler, validation_set_size),
            DataSetSampler.from_sampler(sampler, test_set_size)
        )
