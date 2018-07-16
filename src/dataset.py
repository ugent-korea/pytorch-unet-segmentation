import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from random import randint
from torch.utils.data.dataset import Dataset
from augmentation import *


class SegmentationChallengeData(Dataset):
    def __init__(self, image_path, mask_path):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """

        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))

        # Calculate len
        self.data_len = len(self.mask_arr)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data

        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        # Other approach using torchvision
        """ Other approach with torchvision
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        img_as_np = np.asarray(img_as_img).reshape(1, 512, 512)
        # If there is an operation
        if self.trans == True:
            pass
        # Transform image to tensor
        elif self.trans != True:
            img_as_tensor = self.to_tensor(img_as_np)
        """
        # Get image
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        img_as_np = np.asarray(img_as_img)
        print(img_as_np.dtype)

        # Augmentation
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        flip_img = flip(img_as_np, flip_num)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise}
        noise_det = randint(0, 1)
        if noise_det == 0:
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 10), 0
            noise_img = gaussian_noise(flip_img, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-25, 0), randint(0, 25)
            noise_img = uniform_noise(flip_img, l_bound, u_bound)

        img_as_np = np.expand_dims(img_as_np, axis=0)
        # Augmentation
        img_as_tensor = torch.from_numpy(img_as_np).float()

        # Get mask
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        msk_as_np = np.asarray(msk_as_img)
        msk_as_np = np.expand_dims(msk_as_np, axis=0)

        # Augmentation
        msk_as_tensor = torch.from_numpy(msk_as_np).float()

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


if __name__ == "__main__":

    custom_mnist_from_file_train = SegmentationChallengeData(
        '../data/train/images', '../data/train/masks')
    custom_mnist_from_file_test = SegmentationChallengeData(
        '../data/test/images', '../data/test/masks')

    imag_1 = custom_mnist_from_file_train.__getitem__(2)
    imag_2 = custom_mnist_from_file_test.__getitem__(2)
    print(imag_1)
