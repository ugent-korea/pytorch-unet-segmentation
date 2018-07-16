from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import glob

# Give prediction value from test dataset
class ImagesFromTest(TestSet):

    def __init__(self, image_path, train =True):

        self.totensor = transforms.ToTensor()
        self.img_path = glob.glob(image_path)
		# paths to all images
		self.length = len(self.img_path)
		# number of images

    def __getitem__(self, index):

        im_loc = self.pathways[index]
        img_as_img = Image.open(im_loc)
        img_as_tensor = self.totensor(img_as_img)
        return (img_as_tensor)

    def __len__(self):

		print(self.data_len)
        return self.data_len
