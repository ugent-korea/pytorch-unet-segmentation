from advanced_model import CleanU_Net
#from intuitive_model import CleanU_Net
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
from save_history import *


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
                                    num_workers=16, batch_size=2, shuffle=True)
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=3, batch_size=1, shuffle=True)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=3, batch_size=1, shuffle=True)
    # Dataloader end

    # Model
    model = CleanU_Net(in_channels=1, out_channels=2)
    #model = CleanU_Net()
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # Lo#ss function
    criterion = nn.CrossEntropyLoss()

    # Optimizerd
    optimizer = torch.optim.SGD(model.module.parameters(), lr=0.0005, momentum=0.99)

    # Parameters
    epoch_start = 0
    epoch_end = 5000
    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = "../history/SGD/history_SGD3.csv"
    save_dir = "../history/SGD"
    model_save_dir = "../history/SGD/saved_models"
    image_save_path = "../history/SGD/result_images"
    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        train_model(model, SEM_train_load, criterion, optimizer)
        train_acc, train_loss = get_loss_train(model, SEM_train_load, criterion)
        #train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_acc)
        # Validation
        if (i+1) % 5 == 0:
            val_acc, val_loss = validate_model(
                model, SEM_val_load, criterion, i+1, True, image_save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)
            if (i+1) % 10 == 0:
                save_models(model, model_save_dir, i+1)

            # Test
    '''
    model = copy.deepcopy(model)
    test = test_SEM(model, SEM_test_load, "ih")
    test = Image.fromarray(test)
    test.show()
    '''
