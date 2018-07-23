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
        '../data/test/images/')
    SEM_val = SEMDataVal(
        '../data/val/images', '../data/val/masks')

    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=8, batch_size=12, shuffle=True)
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=16, batch_size=1, shuffle=True)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=8, batch_size=1, shuffle=True)
    model = CleanU_Net(in_channels=1, out_channels=2)
    cuda = torch.cuda.is_available()
    if cuda:
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    else:
        pass
    #model = CleanU_Net(in_channels=1, out_channels=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.1)
    model = train_CE_SEM(model, criterion, optimizer, 5000, SEM_train_load, SEM_val_load, 10)
'''
    model = copy.deepcopy(model)
    test = test_SEM(model, SEM_test_load, "ih")
    test = Image.fromarray(test)
    test.show()
'''
