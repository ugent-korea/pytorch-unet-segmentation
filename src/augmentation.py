import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)
# σ is  the  elasticity  coefficient.


def flip(image, option_value):
    if option_value == 0:
        # vertical
        image == np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image == np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image == np.flip(image, 0)
        image == np.flip(image, 1)
    else:
        pass
        # no effect
    return image


def gaussian_noise(image, mean=0, std=1):
    gaus_noise = np.random.normal(mean, std, image.shape)
    noise_img = image + gaus_noise
    return noise_img


def uniform_noise(image, low=-1, high=1):
    uni_noise = np.random.unifrom(low, high, image.shape)
    noise_img = image + uni_noise
    return noise_img


def brightness(image, value):
    image = image + value
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image
