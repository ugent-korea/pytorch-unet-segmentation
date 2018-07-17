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

Training_MEAN = 0.4911
Training_STDEV = 0.0402


class SEMDataTrain(Dataset):
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
        img_as_img.show()
        img_as_np = np.asarray(img_as_img)

        # Augmentation
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = 3  # randint(0, 3)
        flip_img = flip(img_as_np, flip_num)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise}
        noise_det = randint(0, 1)
        if noise_det == 0:
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            noise_img = add_gaussian_noise(flip_img, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            noise_img = add_uniform_noise(flip_img, l_bound, u_bound)

        # Brightness
        pix_add = randint(-20, 20)
        bright_img = change_brightness(noise_img, pix_add)

        # Elastic distort {0: distort, 1:no distort}
        distort_det = randint(0, 1)
        if distort_det == 0:
            # sigma = 4, alpha = 34
            aug_img, s = add_elastic_transform(bright_img, alpha=34, sigma=4)
        else:
            aug_img = bright_img

        img = Image.fromarray(aug_img)
        img.show()

        print(flip_num, noise_det, distort_det, pix_add)

        # Normalize the image
        norm_img = normalize(aug_img, mean=Training_MEAN, std=Training_STDEV)
        # add additional dimension
        img_as_np = np.expand_dims(norm_img, axis=0)
        # Convert numpy array to tensor
        img_as_tensor = torch.from_numpy(img_as_np).float()

        # Get mask
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)
        # flip the mask with respect to image
        flip_msk = flip(msk_as_np, flip_num)
        if distort_det == 0:
            # sigma = 4, alpha = 34
            aug_msk, _ = add_elastic_transform(flip_msk, alpha=34, sigma=4, seed=s)
            aug_msk = zero_255_image(aug_msk)  # images only with 0 and 255
        else:
            aug_msk = flip_msk

        img2 = Image.fromarray(aug_msk)
        img2.show()

        # add additional dimension
        msk_as_np = np.expand_dims(aug_msk/255, axis=0)
        # Convert numpy array to tensor
        msk_as_tensor = torch.from_numpy(msk_as_np).float()

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


class SEMDataTest(Dataset):

    def __init__(self, image_path, mask_path):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        # paths to all images
        self.length = len(self.img_path)
        # number of images

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path

        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """
        single_image = self.img_path[index]
        img_as_img = Image.open(single_image)

        # Calculate dim1 and dim2 to be overlapped.
        img_dim1 = img_as_img.size[0]
        img_dim2 = img_as_img.size[1]
        overlap_dim1 = img_dim1 - 388
        overlap_dim2 = img_dim2 - 388

        # Convert the image into numpy array
        img_as_numpy = np.asarray(img_as_img)

        # Make 4 cropped image (in numpy array form) using values calculated above
        # Cropped images will also have paddings to fit the model.
        cropped_padded = crop_pad_test(img_as_numpy)
        top_left = cropped_padded[0]
        top_right = cropped_padded[1]
        bottom_left = cropped_padded[2]
        bottom_right = cropped_padded[3]

        # Optional code: see the cropped and padded images
        topleft_img = Image.fromarray(top_left)
        topright_img = Image.fromarray(top_right)
        bottomleft_img = Image.fromarray(bottom_left)
        bottomright_img = Image.fromarray(bottom_right)
        topleft_img.show()
        topright_img.show()
        bottomleft_img.show()
        bottomright_img.show()

        # Convert 4 cropped numpy arrays into tensor
        #img_as_numpy = np.expand_dims(img_as_img, axis=0)
        top_left_tensor = torch.from_numpy(top_left).float()
        top_right_tensor = torch.from_numpy(top_right).float()
        bottom_left_tensor = torch.from_numpy(bottom_left).float()
        bottom_right_tensor = torch.from_numpy(bottom_right).float()

        return (top_left_tensor, top_right_tensor, bottom_left_tensor, bottom_right_tensor)

    def __len__(self):

        return self.data_len


if __name__ == "__main__":

    custom_mnist_from_file_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    custom_mnist_from_file_test = SEMDataTest(
        '../data/test/images/*.png')

    #imag_1 = custom_mnist_from_file_test.__getitem__(0)
    #unique, counts = np.unique(imag_1, return_counts=True)
    #print(dict(zip(unique, counts)))
    imag_2 = custom_mnist_from_file_test.__getitem__(0)
    print(imag_2)
