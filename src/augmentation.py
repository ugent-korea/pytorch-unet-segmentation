import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def add_elastic_transform(image, alpha, sigma, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    from random import randint
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape), seed


def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
        # no effect
    return image


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image)
    return noise_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def zero_255_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def normalize(image, mean, std):
    """
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """

    image = image / 255  # values will lie between 0 and 1.
    image = (image - mean) / std

    return image


def crop_pad_test(image, in_size=572, out_size=388):
    assert out_size*2 >= in_size, "Whole image cannot be expressed with 4 crops"
    img_height, img_width = image.shape[0], image.shape[1]
    l_top = image[:out_size, :out_size]
    r_top = image[:out_size, img_width-out_size:]
    l_bot = image[img_height-out_size:, :out_size]
    r_bot = image[img_height-out_size:, img_width-out_size:]
    print(l_top.shape)
    print(r_top.shape)
    l_top_padded = np.pad(l_top, 92, mode='symmetric')
    r_top_padded = np.pad(r_top, 92, mode='symmetric')
    l_bot_padded = np.pad(l_bot, 92, mode='symmetric')
    r_bot_padded = np.pad(r_bot, 92, mode='symmetric')
    stacked_img = np.stack((l_top_padded, r_top_padded, l_bot_padded, r_bot_padded))
    return stacked_img


def padding():
    pass


if __name__ == "__main__":
    from PIL import Image
    a = Image.open("0.png")
    a.show()
    a = np.array(a)
    croped = crop_pad_test(a)
    a_1, s = add_elastic_transform(a, alpha=34, sigma=4)
    a_2, s = add_elastic_transform(a, alpha=34, sigma=4, seed=s)
    a_3, s = add_elastic_transform(a, alpha=34, sigma=4, seed=s)
    a_11 = Image.fromarray(croped[0])
    a_22 = Image.fromarray(croped[1])
    a_33 = Image.fromarray(croped[2])
    a_44 = Image.fromarray(croped[3])
    a_11.show()
    a_22.show()
    a_33.show()
    a_44.show()
