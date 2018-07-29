from advanced_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from modules import *
from save_history import *


if __name__ == "__main__":
    # Dataset begin
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')

    # TO DO: finish test data loading
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
                                    num_workers=3, batch_size=1, shuffle=False)

    # Dataloader end

    # Model
    model = CleanU_Net(in_channels=1, out_channels=2)
    #model = CleanU_Net()
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.module.parameters(), lr=0.001)

    # Parameters
    epoch_start = 0
    epoch_end = 2000

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = "../history/RMS/history_RMS3.csv"
    save_dir = "../history/RMS"

    # Saving images and models directories
    model_save_dir = "../history/RMS/saved_models3"
    image_save_path = "../history/RMS/result_images3"

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, SEM_train_load, criterion, optimizer)
        train_acc, train_loss = get_loss_train(model, SEM_train_load, criterion)

        #train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i+1) % 5 == 0:
            val_acc, val_loss = validate_model(
                model, SEM_val_load, criterion, i+1, True, image_save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i+1) % 100 == 0:  # save model every 10 epoch
                save_models(model, model_save_dir, i+1)

"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           SEM_test_load, 440, "../history/RMS/result_images_test")
"""
