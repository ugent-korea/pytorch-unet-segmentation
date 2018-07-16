from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import glob

# Give prediction value from test dataset
class ImagesFromTest(TestSet):

    def __init__(self, image_path):

        '''
        Arguments
            image_path = path where images are located
        '''

        self.totensor = transforms.ToTensor()
        self.img_path = glob.glob(image_path)
		# paths to all images
		self.length = len(self.img_path)
		# number of images

    def __getitem__(self, index):

        single_image = self.pathways[index]
        img_as_img = Image.open(im_loc)
        img_as_tensor = self.totensor(img_as_img)
        return img_as_tensor

    def __len__(self):

        return self.data_len

# Experimenting
if __name__ == 'main':

    custom_mnist_test = ImagesFromTest('../data/test/images/*.png')

    test_image = custom_mnist_test.__getitem__()
    print(test_image)
