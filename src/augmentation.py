import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma : σ is an elasticity coefficient
    Return :
        image : elastically transformed numpy array of image
    """
    if random_state is None:
        random_state = numpy.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[1]), numpy.arange(shape[0]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)
# σ is  the  elasticity  coefficient.


def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer that
    Return :
        image : numpy array of flipped image
    """
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
        # no effect
    return image

def gaussian_noise(image, mean=0, std=1):
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceiling_flooring(image)
    return noise_img


def uniform_noise(image, low=-10, high=10):
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceiling_flooring(image)
    return noise_img


def brightness(image, value):
    image = image.astype("int16")
    image = image + value
    image = ceiling_flooring(image)
    return image

def ceiling_flooring(image) :
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image
