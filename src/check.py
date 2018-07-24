import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.segmentation import random_walker
import glob
def accuracy_check(destination) :
    list = glob.glob(destination)
    accuracy = 0
    average = 0
    prediction = np.zeros((512,512))
    prediction = prediction + 255
    for i in range(len(list)) :
        mask = np.array(Image.open(list[i]))
        print(mask.shape)
        compare = np.equal(mask,prediction)
        compare = compare.flatten()
        accuracy = 0
        for element in compare :
            if element ==  True:
                accuracy +=1
        average += accuracy/len(mask.flatten())
    return average/len(list)


a = accuracy_check('../data/train/masks/*.png')
print(a)
