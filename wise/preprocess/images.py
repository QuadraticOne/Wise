import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color
from math import ceil
from random import randint
from scipy.misc import imresize


def load_image(path):
    """
    String -> numpy.ndarray
    Load an image into a tensor form.
    """
    image = mpimg.imread(path)
    if len(image.shape) == 3:
        return image[:, :, :3]
    else:
        return image


def show_image(image_tensor, coloured=True):
    """
    numpy.ndarray -> Bool? -> ()
    Show the coloured image described by a tensor.
    """
    if coloured:
        plt.imshow(image_tensor)
    else:
        plt.imshow(image_tensor, cmap='gray')
    plt.show()


def to_greyscale(image_tensor):
    """
    numpy.ndarray -> numpy.ndarray
    Convert a coloured image to greyscale.
    """
    return color.rgb2gray(image_tensor)


def is_colour(image_tensor):
    """
    numpy.ndarray -> Bool
    Determine whether the image is a colour image
    or a greyscale one.
    """
    return len(image_tensor.shape) == 3


def resize_image(w, h, image_tensor):
    """
    Int -> Int -> numpy.ndarray -> numpy.ndarray
    Resize the image using bilinear interpolation
    to the desired size.
    """
    new_size = (h, w, 3) if is_colour(
        image_tensor) else (h, w)
    return imresize(image_tensor, new_size)


def dataset_from_image(image_tensor, n_samples, output_size,
        output_io, sub_image_size, name):
    """
    numpy.ndarray -> Int -> (Int, Int) -> IO
        -> ((Int, Int) -> ((Int, Int), (Int, Int)))
        -> String -> ()
    Create a dataset of images by taking random sections
    of a larger image, resizing them to the desired output
    size, and saving them using the provided IO object.
    The sub image size is defined in terms of the coordinates
    of the top left, and its height and width, which in turn
    are a function of the image size.
    """
    file_name = '{}-{}'.format(name, '{}')
    image_shape = image_tensor.shape[:2]
    for i in range(n_samples):
        ((top, left), (height, width)) = \
            sub_image_size(image_shape)
        if len(image_tensor.shape) == 3:
            sub = image_tensor[
                top:top + height, left:left + width, :]
        else:
            sub = image_tensor[
                top:top + height, left:left + width]
        sub = resize_image(output_size[0], output_size[1], sub)
        output_io.save_image(sub, file_name.format(i))


def squares_from_image(image_tensor, n_samples, output_size,
        sub_image_size_range, output_io, name):
    """
    numpy.ndarray -> Int -> (Int, Int) -> (Float, Float)
        -> IO -> String -> ()
    Create a dataset of squares from a larger image by
    taking random patches of it.  The size and position
    of each image is determined as a function of the image
    shape, where the maximum and minimum side lengths are
    given as a proportion of the shortest edge of the large
    image.
    """
    size_min, size_max = sub_image_size_range
    def sub_image_size(image_shape):
        h, w = image_shape
        if h < w:
            side_length = randint(
                ceil(h * size_min), ceil(h * size_max))
        else:
            side_length = randint(
                ceil(w * size_min), ceil(w * size_max))

        top = randint(0, h - side_length - 1)
        left = randint(0, w - side_length - 1)
        return (top, left), (side_length, side_length)

    dataset_from_image(image_tensor, n_samples, output_size,
        output_io, sub_image_size, name)


def squares_from_folder(input_io, output_io, samples_per_image,
        sub_image_size_range, output_shape, greyscale=False):
    """
    IO -> IO -> Int -> (Float, Float) -> (Int, Int) -> Bool? -> ()
    Create a number of smaller images from each image in a
    directory, where each subimage is chosen randomly and
    squares are taken from it at random.
    """
    i = 0
    for path in input_io.all_files(remove_extensions=False):
        img = input_io.restore_image(path)
        if img is None:
            continue
        if greyscale:
            img = to_greyscale(img)
        squares_from_image(img, samples_per_image, output_shape,
            sub_image_size_range, output_io, str(i))
        i += 1
