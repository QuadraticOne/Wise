import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color
from scipy.misc import imresize


def load_image(path):
    """
    String -> numpy.ndarray
    Load an image into a tensor form.
    """
    image = mpimg.imread(path)
    return image[:, :, :3]


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
