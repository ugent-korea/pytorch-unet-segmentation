import numpy as np
from PIL import Image
import glob


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
        img_opened = Image.open(image)
        # get the total number of pixels
        total_pixels = img_opened.size[0] * img_opened.size[1]
        img_asarray = np.asarray(img_opened)
        img_asarray = img_asarray / 255
        img_asarray = img_asarray.sum()
        img_asarray = img_asarray / total_pixels
        mean_sum += img_asarray

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
    sq_sum = 0

    for image in all_images:
        img_opened = Image.open(image)
        total_pixels = img_opened.size[0] * img_opened.size[1]
        img_asarray = np.asarray(img_opened)
        img_asarray = img_asarray / 255
        img_asarray = img_asarray.sum()
        img_asarray = img_asarray / total_pixels

        square_diff = (img_asarray - mean_value) ** 2
        sq_sum += square_diff

    stdev = (sq_sum / num_images) ** (1/2)

    return stdev


# Experimenting
if __name__ == '__main__':
    image_path = '../data/train/images'

    print('for training images,')
    print('mean:', Mean(image_path))
    print('stdev:', StandardDeviation(image_path))
