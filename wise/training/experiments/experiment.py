from time import time, sleep
from wise.util.io import IO
from os import listdir
from os.path import isfile, join
from traceback import format_tb
from sys import exc_info


class Experiment:
    """
    Defines a class which can run an experiment, and exposes methods for
    easily logging experiment data when run.
    """

    def __init__(self, log_folder, create_folder_if_missing=False):
        """
        String -> Experiment
        Initialise an instance of Experiment by providing a path to a directory
        in which the experiment files will be stored.
        """
        self.log_folder = log_folder
        self.io = IO(log_folder, create_folder_if_missing)

    def log_experiment(self, file_name):
        """
        String -> ()
        Perform the experiment and log its details in the log directory.  If
        the name of the file already exists, an identifier will be added to
        the end.
        """
        self.io._create_dirs_for_path(file_name)
        jsons = set(self._get_jsons_in_directory(file_name))
        path_without_identifier = file_name.split('/')[-1] + '-'
        i = 0
        while path_without_identifier + str(i) in jsons:
            i += 1
        
        data = self._get_experiment_data()
        self.io.save_json(data, file_name + '-' + str(i))

    def _get_experiment_data(self):
        """
        () -> Dict
        """
        data = {}
        data['start_time_unix'] = time()

        try:
            results = self.run_experiment()
            results['success'] = True
            data['results'] = results
        except Exception as e:
            _, _, traceback = exc_info()
            data['results'] = {
                'success': False,
                'error': str(e),
                'stack_trace': format_tb(traceback)[0]
            }

        data['end_time_unix'] = time()
        data['duration_seconds'] = data['end_time_unix'] - data['start_time_unix']
        return data

    def run_experiment(self):
        """
        () -> Dict
        """
        raise NotImplementedError()

    def _get_jsons_in_directory(self, path):
        """
        String -> [String]
        Return a list of the JSON files in the given directory, without their
        file extension.
        """
        leaf_directory = '/'.join(path.split('/')[:-1]) + '/' \
            if '/' in path else ''
        full_path = self.log_folder + leaf_directory
        files = [f for f in listdir(full_path) if isfile(join(full_path, f))]
        jsons = [f.replace('.json', '') for f in files if '.json' in f]
        return jsons