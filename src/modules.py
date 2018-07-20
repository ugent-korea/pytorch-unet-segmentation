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
import torch.nn as nn
import os


def train_CE_SEM(model, criterion, epoch, img_folder, mask_folder, num_workers=4, batch_size=1, learning_rate=0.01):
    SEM_train = SEMDataTrain(img_folder, mask_folder, in_size=572, out_size=388)
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train, num_workers=num_workers,
                                    batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("initializing training!")
    for i in range(epoch):
        print("epoch:", i)
        for j, (images, masks) in enumerate(SEM_train_load):
            print("image:", j)
            images = Variable(images)
            masks = Variable(masks)  # .view(-1).contiguous()
            outputs = model(images)  # .permute(1, 2, 3, 0).view(-1, 2).contiguous()
            loss = criterion(outputs, masks)
            print(loss)
            optimizer.zero_grad()
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

    for i, (images) in enumerate(SEM_test_load):
        print(images)
        stacked_img = torch.Tensor([])
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0))
            output = model(image)
            print(output)
            #output = softmax(output, dim=1)
            print("size", output.size())
            output = torch.argmax(output, dim=1).float()
            print("size", output.size())
            stacked_img = torch.cat((stacked_img, output))

        div_arr = division_array(388, 2, 2, 512, 512)
        print(stacked_img.size())
        img_cont = image_concatenate(polarize(stacked_img.data.numpy()), 2, 2, 512, 512)
        final_img = (img_cont*255/div_arr)
        print(final_img)
        final_img = final_img.astype("uint8")

        # Save images in every epoch
        desired_path = folder_to_save + '/' + str(i)
        # Create the path if it does not exist
        if not os.path.exists(desired_path):
            os.makedirs(desired_path)

        break
    return final_img


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img
