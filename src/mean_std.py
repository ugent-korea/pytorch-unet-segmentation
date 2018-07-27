import numpy as np
from PIL import Image
import glob


def normalize_image(image):
    """
    Args:
        image : a string of name of image file
    Return:
        image_asarray : numpy array of the image
                        that is normalized by being divided by 255
    """

    img_opened = Image.open(image)
    img_asarray = np.asarray(img_opened)
    img_asarray = img_asarray / 255

    return img_asarray


def find_mean(image_path):
    """
    Args:
        image_path : pathway of all images
    Return :
        mean : mean value of all the images
    """
    all_images = glob.glob(str(image_path) + str("/*"))
    num_images = len(all_images)
    mean_sum = 0

    for image in all_images:
        img_asarray = normalize_image(image)
        individual_mean = np.mean(img_asarray)
        mean_sum += individual_mean

    # Divide the sum of all values by the number of images present
    mean = mean_sum / num_images

    return mean


def find_stdev(image_path):
    """
    Args:
        image_path : pathway of all images
    Return :
        stdev : standard deviation of all pixels
    """
    # Initiation
    all_images = glob.glob(str(image_path) + str("/*"))
    num_images = len(all_images)

    # Recall mean value from function above: def Mean(path)
    mean_value = find_mean(image_path)
    std_sum = 0

    for image in all_images:
        img_asarray = normalize_image(image)
        individual_stdev = np.std(img_asarray)
        std_sum += individual_stdev

    std = std_sum / num_images

    return std


# Experimenting
if __name__ == '__main__':
    image_path = '../data/train/images'

    print('for training images,')
    print('mean:', find_mean(image_path))
    print('stdev:', find_stdev(image_path))
