import numpy as np
from PIL import Image
import glob

def Mean(image):
    """
    Args:
        image : numpy array of image
    Return :
        mean : mean value of all the pixels
    """

    # get the total number of pixels
    total_pixels = image_opened.shape[0] * image_opened.shape[1]

    img_as_array = img_as_array / 255
    img_as_array = img_as_array.sum()

    # Divide the sum of all values by the number of values present
    mean = img_as_array / total_pixels

    return mean

def StandardDeviation(image):
    """
    Args:
        image : numpy array of image
    Return :
        stdev : standard deviation of all pixels
    """

    # Recall mean value from function above: def Mean(path)
    mean_value = Mean(img_path)

    total_pixels = image_opened.shape[0] * image_opened.shape[1]
    img_as_array = np.asarray(image_opened)

    img_as_array = img_as_array / 255

    square_sum = 0  # sum of (x - mean_value) ** 2

    for array in img_as_array:
        for element in array:
            ith_square = (element - mean_value) ** 2
            square_sum += ith_square

    # Finishing the Standard Deviation formula
    stdev = square_sum / total_pixels
    stdev = (stdev) ** 0.5

    return stdev
