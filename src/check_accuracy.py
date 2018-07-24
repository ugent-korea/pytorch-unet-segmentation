import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.segmentation import random_walker

def accuracy_check(mask, prediction) :
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)) :
            item = np.array(item)
        elif 'torch' in str(type(item)) :
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())
