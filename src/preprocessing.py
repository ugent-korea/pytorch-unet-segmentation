import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import glob

class preprocess(dataset):

    def __init__(self, image_path, train = True):
        '''
        Arguments:
            image_path = path where images are located
            train = if True, the dataset is a training set. Else, the dataset is a test set
        '''
        self.img_path = glob.glob(image_path)
        # Path that contains all the images

        self.length = len(self.img_path)
        # Number of images in the path
