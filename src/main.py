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
from loss_functions import CELoss


if __name__ == "__main__":
    # Dataset begin
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/')
    SEM_val = SEMDataVal(
        '../data/val/images', '../data/val/masks')
    # Dataset end

    # Dataloader begins
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=4, batch_size=8, shuffle=True)
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=3, batch_size=1, shuffle=True)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=3, batch_size=1, shuffle=True)
    # Dataloader end

    # Model
    model = CleanU_Net(in_channels=1, out_channels=2)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = CELoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    # Parameters
    epoch_start = 0
    epoch_end = 100

    # Train
    for i in range(epoch_start, epoch_end):
        train_loss = train_model(model, SEM_train_load, criterion, optimizer)
        train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i), 'Train loss:', train_loss)
        # Validation
        if i%5 == 0:
            pass
            # Test
    '''
    model = copy.deepcopy(model)
    test = test_SEM(model, SEM_test_load, "ih")
    test = Image.fromarray(test)
    test.show()
    '''
