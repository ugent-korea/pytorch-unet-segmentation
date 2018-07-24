import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.segmentation import random_walker
mask = '28.png'
#prediction = np.array(Image.open('test3.png'))
prediction1 = np.zeros((512,512))

def accuracy_check(mask, prediction) :
    if type(mask) is  str :
        mask = np.array(Image.open(mask))
    elif type(mask) is 'PIL.PngImagePlugin.PngImageFile' :
        mask = np.array(prediction)
    if type(prediction) is  str :
        mask = np.array(Image.open(mask))
    elif type(prediction) is 'PIL.PngImagePlugin.PngImageFile' :
        mask = np.array(prediction)

    compare = np.equal(mask,prediction)
    compare = compare.flatten()
    accuracy = 0
    for element in compare :
        if element ==  True:
            accuracy +=1

    return accuracy/len(compare)
