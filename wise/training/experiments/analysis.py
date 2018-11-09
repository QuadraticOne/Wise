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

    def filter_experiments(self, predicate):
        """
        (Dict -> Bool) -> [Dict]
        Return a list of all experiments which match the given predicate.
        """
        filtered_list = []
        for path in self.experiment_paths:
            self._load_experiment_if_unloaded(path)
            experiment = self.get_by_path(path)
            if predicate(experiment):
                filtered_list.append(experiment)
            self._unload_experiment(path)
        return filtered_list

    def extract_results(self, result_getters, experiments=None):
        """
        [(Dict -> Object)] -> [Dict]? -> [[Object]]
        Get the specified result from each experiment in the given list.  If
        no list is provided, this will apply to all experiments.
        """
        dicts = experiments if experiments is not None else \
            (self.get_by_path(path) for path in self.experiment_paths)
        return [
            [getter(d) for getter in result_getters] for d in dicts
        ]

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
        String -> ()
        Checks if the experiment has been loaded or not, and loads it if not.
        """
        if self.experiments[path] is None:
            with open(path, 'r') as f:
                self.experiments[path] = loads(f.read())

    def _unload_experiment(self, path):
        """
        String -> ()
        Unloads the experiment data from memory.
        """
        self.experiments[path] = None

    def get_by_path(self, path):
        """
        String -> Dict
        Return the experiment data for the given experiment name.
        """
        self._load_experiment_if_unloaded(path)
        return self.experiments[path]


def dict_contains_key(nested_key):
    """
    String -> (Dict -> Bool)
    Return a function which takes a dictionary and determines whether it
    contains the nested key.  Nested keys are strings which allow indexing
    over nested dictionaries, by separating the keys at each level with
    a full stop as if it were being accessed in plain JavaScript.  
    """
    unnested_keys = nested_key.split('.')
    return lambda _d: _dict_contains_key_from_list(unnested_keys, _d)


def _dict_contains_key_from_list(keys, d):
    """
    [String] -> Dict -> Bool
    Given a list of nested keys and a dictionary, determine whether or
    not that dictionary contains the leaf element produced by indexing
    all the keys.
    """
    if len(keys) == 0:
        return True
    else:
        return dict_contains_key_from_list(keys[1:], d[keys[0]]) \
            if keys[0] in d else False


def predicate_on_dictionary_key(nested_key, predicate):
    """
    String -> (Object -> Bool) -> (Dict -> Bool)
    Return a function which takes a dictionary and returns the result
    of a predicate which takes a value as indexed by the given nested key.
    If the dictionary does not contain the key, False will be returned
    by default.
    """
    unnested_keys = nested_key.split('.')
    return lambda d: _predicate_on_key_from_list(unnested_keys, predicate, d)


def _predicate_on_key_from_list(keys, predicate, d):
    """
    [String] -> (Object -> Bool) -> Dict -> Bool
    Given a list of nested keys, a predicate, and a dictionary, determine
    whether or not the leaf element produced by indexing all the keys
    satisfies the predicate.  If any of the keys are not in the dictionary
    this will return False.
    """
    if len(keys) == 0:
        return predicate(d)
    else:
        return _predicate_on_key_from_list(keys[1:], predicate, d[keys[0]]) \
            if keys[0] in d else False
