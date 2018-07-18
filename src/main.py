from dynamic_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
from torch.utils.data.dataset import Dataset

if __name__ == "__main__":
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')

    SEM_test = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=8, batch_size=1, shuffle=True)
    SEM_train = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=8, batch_size=1, shuffle=True)

    model = CleanU_Net(1)

    for i, (images, labels) in enumerate(SEM_train):
        images = Variable(images)
        labels = Variable(labels)
        print("input size", images.size())
        print(labels.size())
        output = model(images)
        print(output.shape)
        break

    for i, (images, labels) in enumerate(SEM_test):
        images = Variable(images)
        labels = Variable(labels)
        print("input size", images.size())
        output = model(images)
        print(output.shape)
        break
