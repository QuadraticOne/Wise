from os.path import isdir
from os import makedirs
from pickle import dump, load
import tensorflow as tf


class IO:
    """
    Creates objects for quickly and easily handling IO operations
    in a directory.
    """
    
    def __init__(self, file_path, create_if_missing=False):
        self.file_path = file_path
        self._validate_file_path()
        
        if not self._dir_exists():
            if create_if_missing:
                self._create_dir()
            else:
                raise FileNotFoundError('File path not found: "{}".'
                    .format(self.file_path))

    def _validate_file_path(self):
        """
        () -> ()
        Throws exception if the file path is not valid.  A valid file path
        has a forward slash at the end and contains no full stops.
        """
        ends_in_slash = self.file_path[-1] == '/'
        contains_dot = '.' in self.file_path
        if not ends_in_slash:
            raise ValueError('Invalid file path: should end in "/".')
        if contains_dot:
            raise ValueError('Invalid file path: should not contain ".".')

    def _dir_exists(self, inner_dir=None):
        """
        String? -> Bool
        Check if the directory exists, but do nothing.
        """
        return isdir(self.file_path + (inner_dir or ''))

    def _create_dir(self, inner_dir=None):
        """
        String? -> ()
        Create the home directory, assuming it does not already exist.
        """
        makedirs(self.file_path + (inner_dir or ''))

    def _extend(self, path, extension=''):
        """
        String -> String -> String
        Add the path, and then the extension, onto the end of the home directory.
        """
        return self.file_path + path + extension

    def save_object(self, obj, path):
        """
        Object -> String -> ()
        """
        self._create_dirs_for_path(path)
        with open(self._extend(path, '.obj'), 'wb') as f:
            dump(obj, f)

    def restore_object(self, path):
        """
        String -> Object
        """
        obj = None
        with open(self._extend(path, '.obj'), 'rb') as f:
            obj = load(f)
        return obj

    def save_session(self, session, path, variables=None):
        """
        tf.Session -> String -> [tf.Variable]? -> ()
        """
        self._create_dirs_for_path(path)
        saver = tf.train.Saver(save_relative_paths=True, var_list=variables)
        saver.save(session, self._extend(path + '/' + path))

    def restore_session(self, session, path, variables=None):
        """
        tf.Session -> String -> [tf.Variable]? -> ()
        """
        saver = tf.train.Saver(save_relative_paths=True, var_list=variables)
        saver.restore(session, self._extend(path + '/' + path))

    def _create_dirs_for_path(self, path):
        """
        String -> ()
        """
        dir_path = self._remove_file_from_path(path)
        if not self._dir_exists(dir_path):
            self._create_dir(dir_path)

    def _remove_file_from_path(self, path):
        """
        String -> String
        """
        return '/'.join(path.split('/')[:-1])
