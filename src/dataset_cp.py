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
from mean_std import *

Training_MEAN = 0.4911
Training_STDEV = 0.1658


class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
        self.mask_arr = glob.glob(str(mask_path) + "/*")
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size, self.out_size = in_size, out_size
        # Calculate len
        self.data_len = len(self.mask_arr)
        # calculate mean and stdev

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        """
        # GET IMAGE
        """
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)

        # Augmentation
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        img_as_np = flip(img_as_np, flip_num)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        if randint(0, 1):
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

        # Brightness
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)

        # Elastic distort {0: distort, 1:no distort}
        sigma = randint(6, 12)
        # sigma = 4, alpha = 34
        img_as_np, seed = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20)

        # Crop the image
        img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
        y_loc, x_loc = randint(0, img_height-self.out_size), randint(0, img_width-self.out_size)
        img_as_np = cropping(img_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)

        # Pad the image
        img_as_np = add_padding(img_as_np, in_size=self.in_size,
                                out_size=self.out_size, mode="symmetric")

        # Normalize the image
        img_as_np = normalization(img_as_np, max=1, min=0)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        # msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)

        # flip the mask with respect to image
        msk_as_np = flip(msk_as_np, flip_num)

        # elastic_transform of mask with respect to image

        # sigma = 4, alpha = 34, seed = from image transformation
        msk_as_np, _ = add_elastic_transform(
            msk_as_np, alpha=34, sigma=sigma, seed=seed, pad_size=20)
        msk_as_np = approximate_image(msk_as_np)  # images only with 0 and 255

        # Crop the mask
        msk_as_np = msk_as_np[y_loc:y_loc+self.out_size, x_loc:x_loc+self.out_size]

        """
        # Sanity Check for mask
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


class SEMDataVal(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks
        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.data_len = len(self.mask_arr)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """
        single_image = self.image_arr[index]
        img_as_img = Image.open(single_image)
        # img_as_img.show()
        # Convert the image into numpy array
        img_as_numpy = np.asarray(img_as_img)

        # Make 4 cropped image (in numpy array form) using values calculated above
        # Cropped images will also have paddings to fit the model.

        img_as_numpy = multi_cropping(img_as_numpy,
                                      crop_size=self.out_size,
                                      crop_num1=2, crop_num2=2)
        img_as_numpy = multi_padding(img_as_numpy, in_size=self.in_size,
                                     out_size=self.out_size, mode="symmetric")

        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []

        for array in img_as_numpy:

            # SANITY CHECK: SEE THE CROPPED & PADDED IMAGES
            #array_image = Image.fromarray(array)

            # Normalize the cropped arrays
            img_to_add = normalization(array, max=1, min=0)
            # img_as_numpy = np.expand_dims(img_as_numpy, axis=0)
            # img_as_tensor = torch.from_numpy(img_as_numpy).float()
            # Convert normalized array into tensor
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        # msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)
        # Normalize mask to only 0 and 1

        msk_as_np = multi_cropping(msk_as_np,
                                   crop_size=self.out_size,
                                   crop_num1=2, crop_num2=2)

        msk_as_np = msk_as_np/255

        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        original_msk = torch.from_numpy(np.asarray(msk_as_img))
        return (img_as_tensor, msk_as_tensor, original_msk)

    def __len__(self):

        return self.data_len


class SEMDataTest(Dataset):

    def __init__(self, image_path, in_size=572, out_size=388):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks

        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size = in_size
        self.out_size = out_size
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """

        single_image = self.image_arr[index]
        img_as_img = Image.open(single_image)
        # img_as_img.show()
        # Convert the image into numpy array
        img_as_numpy = np.asarray(img_as_img)

        # Make 4 cropped image (in numpy array form) using values calculated above
        # Cropped images will also have paddings to fit the model.

        img_as_numpy = multi_cropping(img_as_numpy,
                                      crop_size=self.out_size,
                                      crop_num1=2, crop_num2=2)
        img_as_numpy = multi_padding(img_as_numpy, in_size=self.in_size,
                                     out_size=self.out_size, mode="symmetric")

        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []

        for array in img_as_numpy:

            # SANITY CHECK: SEE THE CROPPED & PADDED IMAGES
            array_image = Image.fromarray(array)
            # array_image.show()

            # Normalize the cropped arrays
            img_as_numpy = normalize(array, mean=Training_MEAN, std=Training_STDEV)
            # img_as_numpy = np.expand_dims(img_as_numpy, axis=0)
            # img_as_tensor = torch.from_numpy(img_as_numpy).float()
            # Convert normalized array into tensor
            processed_list.append(img_as_numpy)

        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.
        return img_as_tensor

    def __len__(self):

        return self.data_len


if __name__ == "__main__":

    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')
    SEM_val = SEMDataVal('../data/val/images', '../data/val/masks')

    imag_1, msk = SEM_train.__getitem__(0)

    print(msk)
