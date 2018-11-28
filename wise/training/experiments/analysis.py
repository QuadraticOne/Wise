from os import listdir
from os.path import join, isdir
from json import loads
import matplotlib.pyplot as plt


class ResultsFilter:
    """
    Class for filtering the results of multiple experiments and plotting them.
    """

    BASE_PLOT_RADIUS = 25
    PLOT_COLOURS = 'brcym'

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
        no list is provided, this will apply to all experiments.  The inner
        list contains the result of each getter for a single datum, while the
        outer list contains one of these for each datum.
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

    def plot_results(self, x=None, y=None, z=None, group=None, radius=lambda _: 1,
            x_label='x', y_label='y', z_label='z'):
        """
        (Dict -> Float)? -> (Dict -> Float)? -> (Dict -> Float)? -> (Dict -> String)?
            -> (Dict -> Int)? -> String? -> String? -> String? -> ()
        Plot the results, providing functions which extract the properties of
        each point from each point in the data set.
        """
        groups = {}
        if group is None:
            groups['data'] = []
            for path in self.experiment_paths:
                self._load_experiment_if_unloaded(path)
                experiment = self.get_by_path(path)
                groups['data'].append(experiment)
                self._unload_experiment(path)
        else:
            for path in self.experiment_paths:
                self._load_experiment_if_unloaded(path)
                experiment = self.get_by_path(path)
                _group = group(experiment)
                if _group in groups:
                    groups[_group].append(experiment)
                else:
                    groups[_group] = [experiment]
                self._unload_experiment(path)

        if z is None:
            self._plot_results_2d(x, y, groups, radius, x_label, y_label)
        else:
            self._plot_results_3d(x, y, z, groups, radius, x_label, y_label, z_label)

    def _plot_results_2d(self, x, y, groups, radius, x_label, y_label):
        """
        Plot the data set in two dimensions.  See the documentation of
        `plot_results` for argument types, except `groups`, which is a dictionary.
        """
        colour_index = 0
        legend_scatters = []
        legend_names = []

        for label, group_experiments in groups.items():
            values = self.extract_results([x, y, radius], group_experiments)
            xs, ys, radii = [], [], []
            for point in values:
                xs.append(point[0])
                ys.append(point[1])
                radii.append(point[2] * point[2] * ResultsFilter.BASE_PLOT_RADIUS)
            legend_scatters.append(plt.scatter(xs, ys,
                c=ResultsFilter.PLOT_COLOURS[colour_index], s=radii, label=label))
            legend_names.append(label)
            colour_index += 1

        plt.legend(legend_scatters, legend_names)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def _plot_results_3d(self, x, y, z, groups, radius, x_label, y_label, z_label):
        """
        Plot the data set in three dimensions.  See the documentation of
        `plot_results` for argument types, except `groups`, which is a dictionary.
        """
        raise NotImplementedError('3D plotting not yet implemented')


def dict_contains_key(nested_key):
    """
    String -> (Dict -> Bool)
    Return a function which takes a dictionary and determines whether it
    contains the nested key.  Nested keys are strings which allow indexing
    over nested dictionaries, by separating the keys at each level with
    a full stop as if it were being accessed in plain JavaScript.  
    """
    unnested_keys = nested_key.split('.')
    return lambda d: _dict_contains_key_from_list(unnested_keys, d)


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


def map_on_dictionary_key(nested_key, f, expect_list_at_leaf=False):
    """
    String -> (Object -> Object) -> (Dict -> Object)
    Return a function which indexes a dictionary, applies a function to
    the value stored in that index, and returns that value.

    Where any of the keys are lists, the result will be compiled from
    each element into a list of results.  If the leaf node is expected
    to be a list and this behaviour should be deactivated for the leaf
    node, set `expect_list_at_leaf` to True.
    """
    unnested_keys = nested_key.split('.')
    return lambda d: _map_on_key_from_list(unnested_keys, f, d,
        expect_list_at_leaf=expect_list_at_leaf)


def _map_on_key_from_list(keys, f, d, expect_list_at_leaf=False):
    """
    [String] -> (Object -> Object) -> Dict -> Object
    Given a list of nested keys, a transfer function, and a dictionary,
    return the result of passing the value referenced by applying the
    keys recursively to the dictionary through the transfer function.
    If the dictionary does not contain that key, returns None.

    Where any of the keys are lists, the result will be compiled from
    each element into a list of results.  If the leaf node is expected
    to be a list and this behaviour should be deactivated for the leaf
    node, set `expect_list_at_leaf` to True.
    """
    if len(keys) == 0:
        return f(d)
    else:
        if keys[0] not in d:
            return None

        ignore_mapping = len(keys) == 1 and expect_list_at_leaf
        if type(d[keys[0]]) == type([]) and not ignore_mapping:
            return [_map_on_key_from_list(keys[1:], f, sub_key) \
                for sub_key in d[keys[0]]]
        else: 
            return _map_on_key_from_list(keys[1:], f, d[keys[0]])


def group_by(data, grouping_function):
    """
    [a] -> (a -> Object) -> Dict
    Group the given data based on the value that is returned by
    feeding each datum to a grouping function.
    """
    groups = {}
    for datum in data:
        group = grouping_function(datum)
        if group in groups:
            groups[group].append(datum)
        else:
            groups[group] = [datum]
    return groups
