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
from modules import *
import copy

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

    test = train_CE_SEM(model, criterion, epoch=2, img_folder='../data/train/images',
                        mask_folder='../data/train/masks')

    #test = test_SEM(model, criterion, '../data/test/images/', '../data/test/masks', "ih")
    test = Image.fromarray(test)
    test.show()
