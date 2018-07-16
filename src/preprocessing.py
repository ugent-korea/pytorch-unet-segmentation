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
    total_pixels = image.shape[0] * image.shape[1]

    image = image / 255
    image = image.sum()

    # Divide the sum of all values by the number of values present
    mean = image / total_pixels

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

    total_pixels = image.shape[0] * image.shape[1]

    image = image / 255

    square_sum = 0  # sum of (x - mean_value) ** 2

    for array in image:
        for element in array:
            ith_square = (element - mean_value) ** 2
            square_sum += ith_square

    # Finishing the Standard Deviation formula
    stdev = square_sum / total_pixels
    stdev = (stdev) ** 0.5

    return stdev

# Experimenting
if __name__ == '__main__':
    image_path = '../data/train/images/*.png'

    all_img = glob.glob(image_path)
    mean_sum = 0
    sq_sum = 0

    for img in all_img:
        img_opened = Image.open(img)
        img_asarray = np.asarray(img_opened)
        mean_sum += Mean(img_asarray)

    # mean value for all the images in the designated folder.
    mean_sum = mean_sum / len(all_img)

    for img in all_img:

        img_opened = Image.open(img)
        img_asarray = np.asarray(img_opened)
        img_asarray = img_asarray / 255
        img_asarray = img_asarray.sum()
        total_pixels = img_opened.size[0] * img_opened.size[1]
        img_asarray = img_asarray / total_pixels
        sq_sum += (img_asarray - mean_sum) ** 2
    sq_sum = sq_sum / len(all_img)

    # standard deviation of all the images in the designated folder.
    stdev = sq_sum ** 0.5

    print('for training images,')
    print('standard deviation:', stdev)
    print('mean:', mean_sum)
