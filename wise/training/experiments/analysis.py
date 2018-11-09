from os import listdir
from os.path import join, isdir
from json import loads


class ResultsFilter:
    """
    Class for filtering the results of multiple experiments and plotting them.
    """

    def __init__(self, root_directory, include_sub_directories=False):
        """
        String -> Bool? -> ResultsFilter
        Create a filter allowing quick and easy access to the results of all
        experiments stored in the given directory.  If `include_sub_directories`
        is true then all experiments in folders within the root directory
        will also be available for loading.
        """
        self.root_directory = root_directory
        self.include_sub_directories = include_sub_directories

        self.experiment_paths = None
        self.experiments = None

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        Find all available experiments but do not load any yet.
        """
        self.experiment_paths = self._get_experiments(self.root_directory, True)
        self.experiments = {path: None for path in self.experiment_paths}

    def _get_experiments(self, file_path, force_include_sub_directories):
        """
        String -> Bool -> [String]
        Get a list of all experiments in the directory, if the path points
        to a directory and `include_sub_directories` is set to True, or
        simply return the file name if the path references a file.  Since
        this function is called recursively, an option is included to force
        the inclusion of sub-directories.
        """
        experiments = []
        if isdir(file_path):
            if self.include_sub_directories or force_include_sub_directories:
                for child_file in listdir(file_path):
                    experiments += self._get_experiments(
                        join(file_path, child_file), False)
        else:
            if '.json' in file_path:
                experiments.append(file_path)
        return experiments

    def _load_experiment_if_unloaded(self, path):
        """
        () -> ()
        Checks if the experiment has been loaded or not, and loads it if not.
        """
        if self.experiments[path] is None:
            with open(path, 'r') as f:
                self.experiments[path] = loads(f.read())
