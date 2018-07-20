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
import csv


def train_CE_SEM(model, criterion, optimizer, epoch, data_train, data_val):

    print("initializing training!")

    for i in range(epoch):
        print("epoch:", i+1)
        for j, (images, masks) in enumerate(data_train):
            print("image:", j)
            images = Variable(images)
            masks = Variable(masks)
            print(masks.shape)  # .view(-1).contiguous()
            outputs = model(images)  # .permute(1, 2, 3, 0).view(-1, 2).contiguous()
            loss = criterion(outputs, masks)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
        del outputs
        del images
        del masks

        val_loss = 0
        print("validation")
        for j, (images_v, masks_v) in enumerate(data_val):
            for t in range(images_v.size()[1]):
                print("image:", j)
                image_v = Variable(images_v[:, t, :, :].unsqueeze(0))
                mask_v = Variable(masks_v[:, t, :, :].squeeze(1))  # .view(-1).contiguous()
                print(mask_v.shape)
                output_v = model(image_v)  # .permute(1, 2, 3, 0).view(-1, 2).contiguous()
                val_loss += criterion(output_v, mask_v)
            break
            del output_v
            del images_v
            del masks_v
        print("epoch: {0}/{1}| loss: {2:.4f}".format(i+1, epoch, val_loss/((j+1)*4)))

    with open('train_loss.csv', 'w') as csvfile:
        fieldnames = ['epoch', 'image', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
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
                writer.writerow({'epoch': i+1, 'image': j+1, 'loss': float(loss)})

    return model


def test_SEM(model, data_test,  folder_to_save):

    for i, (images) in enumerate(data_test):

    for i, (images) in enumerate(SEM_test_load):
        print(i)

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
        img_cont = image_concatenate(stacked_img.data.numpy(), 2, 2, 512, 512)
        final_img = (img_cont*255/div_arr)
        print(final_img)
        final_img = final_img.astype("uint8")
        break
    return final_img


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img
