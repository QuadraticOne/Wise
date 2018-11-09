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

        self.experiments = None

    def _get_experiments(self):
        """
        () -> ()
        Populate the experiments field of this object with the names of all
        available experiments in the relevant directories.
        """
        self.experiments = {}
        # TODO: finish implementation

    def _get_experiments(self, file_path):
        pass
