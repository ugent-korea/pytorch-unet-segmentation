import pandas as pd
import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from dataset import *
from torch.nn.functional import softmax


def train_CE_SEM(model, epoch, folder_to_call, num_workers=4, batch_size=1, learning_rate=0.01):
    SEM_train = SEMDataTrain(folder_to_call, in_size=572, out_size=388)
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train, num_workers=num_workers,
                                    batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print("initializing training!")
    for i in range(epoch):
        for j, (images, masks) in enumerate(SEM_train_load):
            images = Variable(images)
            masks = Variable(masks)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Update weights
            optimizer.step()
    return model


def test_SEM(model, loss_function, img_folder, mask_folder, folder_to_save):
    SEM_test = SEMDataTest(img_folder, mask_folder, in_size=572, out_size=388)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test, num_workers=4,
                                    batch_size=1, shuffle=True)
    print(SEM_test_load)
    criterion = loss_function
    stacked_img = torch.Tensor([])
    for i, (images) in enumerate(SEM_test_load):
        print(images)
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0))
            output = model(image)
            output = softmax(output, dim=1)[:, 0, :, :]
            stacked_img = torch.cat((stacked_img, output))
        div_arr = division_array(388, 2, 2, 512, 512)
        print(np.asarray(stacked_img).shape)
        img_cont = image_concatenate(np.asarray(stacked_img), 2, 2, 512, 512)
        final_img = (img_cont/div_arr)*255
        final_img = final_img.astype("uint8")
        break
    return final_img


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return
