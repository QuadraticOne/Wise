from os.path import isdir
from os import makedirs, listdir
from pickle import dump, load
from json import dumps, loads
from scipy.misc import imsave
import tensorflow as tf
import matplotlib.image as mpimg


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

    def save_json(self, json_dict, path):
        """
        Dict -> String -> ()
        Save the data contained in the python dictionary into
        a JSON file.
        """
        with open(self._extend(path, '.json'), 'w') as f:
            f.write(dumps(json_dict))

    def restore_json(self, path):
        """
        String -> Dict
        Restore the data contained in the JSON file into a
        python dictionary.
        """
        with open(self._extend(path, '.json'), 'r') as f:
            obj = loads(f.read())
        return obj

    def save_image(self, image_tensor, path):
        """
        numpy.ndarray -> String -> ()
        Save the given image as a .png file.
        """
        imsave(self._extend(path, extension='.png'), image_tensor)

    def restore_image(self, path):
        """
        String -> numpy.ndarray
        Restore the image contained at the given location
        to an image, represented as a ndarray.  Note that
        the file extension must be included.
        """
        image = mpimg.imread(self._extend(path))
        if len(image.shape) == 3:
            return image[:, :, :3]
        else:
            return image

    def all_files(self, sub_file_path='', include_sub_folders=False,
            remove_extensions=True):
        """
        String? -> Bool? -> Bool? -> [String]
        """
        files = listdir(self.file_path + sub_file_path)
        processed_files = []
        for f in files:
            if not isdir(self.file_path + sub_file_path + f):
                processed_files.append(sub_file_path +
                    (self._remove_file_extension(f) if remove_extensions else f))
            else:
                if include_sub_folders:
                    processed_files += self.all_files(
                        sub_file_path=sub_file_path + f + '/',
                        include_sub_folders=include_sub_folders,
                        remove_extensions=remove_extensions)
        return processed_files

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

    def _remove_file_extension(self, path):
        """
        String -> String
        """
        return path.split('.')[0]
