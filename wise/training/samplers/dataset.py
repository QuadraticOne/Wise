from wise.training.samplers.sampler import Sampler
from wise.util.io import IO
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

    def save(self, directory, file_name):
        """
        String -> String -> ()
        Save the points that make up this data set to a file.  They can be
        restored using DataSetSampler.restore(...).  Do not include a file
        extension in the file name.
        """
        io = IO(directory, create_if_missing=True)
        io.save_object(self.points, file_name)

    @staticmethod
    def restore(directory, file_name):
        """
        String -> String -> DataSetSampler
        Restore a data set sampler that was previously saved into the given
        directory and file name.  Do not include a file extension in the file
        name.
        """
        io = IO(directory, create_if_missing=False)
        data = io.restore_object(file_name)
        return DataSetSampler(data)

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
