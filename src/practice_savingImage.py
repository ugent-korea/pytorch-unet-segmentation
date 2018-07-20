from advanced_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
from torch.utils.data.dataset import Dataset
from torch.nn.functional import softmax, cross_entropy
from modules_draft import *
import copy
import os

if __name__ == "__main__":
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')

    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=8, batch_size=1, shuffle=True)
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=8, batch_size=1, shuffle=True)

    model = CleanU_Net(in_channels=1, out_channels=2)
    criterion = nn.CrossEntropyLoss()

    model = train_CE_SEM(model, criterion, epoch=1, img_folder='../data/train/images',
                         mask_folder='../data/train/masks')

    model = copy.deepcopy(model)
    test = test_SEM(model, criterion, 3, '../data/test/images', '../data/test/masks', "/result_images/")
    test = Image.fromarray(test)
    test.show()
