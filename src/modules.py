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
import os
import time


def train_CE_SEM(model, criterion, optimizer, epoch, data_train, data_val, checkpoint=5):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        epoch: number of epoch
        data_train (DataLoader): training dataset
        data_val (DataLoader): validation dataset
        checkpoint (int): how frequent to save the images and models
    """
    print("initializing training!")
    with open('progress_report.csv', 'w') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(epoch):
            for j, (images, masks) in enumerate(data_train):
                # print("image:", j)
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
                # print(masks.shape)  # .view(-1).contiguous()
                outputs = model(images)  # .permute(1, 2, 3, 0).view(-1, 2).contiguous()
                loss = criterion(outputs, masks)
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
            del outputs
            del images
            del masks

            # calculating training loss
            train_loss = 0
            for dt, (images, masks) in enumerate(data_train):
                with torch.no_grad():
                    # print("image:", j)
                    images = Variable(images.cuda())
                    masks = Variable(masks.cuda())
                    # print(masks.shape)  # .view(-1).contiguous()
                    outputs = model(images)  # .permute(1, 2, 3, 0).view(-1, 2).contiguous()
                    train_loss += criterion(outputs, masks)
                del outputs
                del images
                del masks

            # calculating validation loss
            val_loss = 0
            div_arr = division_array(388, 2, 2, 512, 512)
            for j, (images_v, masks_v) in enumerate(data_val):
                stacked_img = torch.Tensor([]).cuda()
                for t in range(images_v.size()[1]):
                    with torch.no_grad():
                        # print("image:", j)
                        image_v = Variable(images_v[:, t, :, :].unsqueeze(0).cuda())
                        mask_v = Variable(masks_v[:, t, :, :].squeeze(1).cuda())
                        # print(mask_v.shape)
                        output_v = model(image_v)
                        val_loss += criterion(output_v, mask_v)
                        if (i+1) % checkpoint == 0:

                            output_v = torch.argmax(output_v, dim=1).float()
                            stacked_img = torch.cat((stacked_img, output_v))
                        else:
                            pass
                if (i+1) % checkpoint == 0:

                    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), 2, 2, 512, 512)
                    img_cont = (img_cont*255)/div_arr
                    img_cont = img_cont.astype('uint8')

                    img_cont = Image.fromarray(img_cont)
                    # organize images in every epoch
                    desired_path = 'result_images/epoch_' + str(i+1) + '/'

                    # Create the path if it does not exist
                    if not os.path.exists(desired_path):
                        os.makedirs(desired_path)

                    # Save Image!
                    export_name = 'test' + str(j) + '.png'
                    img_cont.save(desired_path + export_name)
                    torch.save(model, "saved_models/model_epoch_{0}.pwf".format(i+1))

                else:

                    pass
                del output_v
                del images_v
                del masks_v
                del image_v
                del mask_v
            print("epoch: {0}/{1}|train_loss: {2:.10f}|validation_loss: {3:.10f}".format(i +
                                                                                         1, epoch, train_loss/(dt+1), val_loss/((j+1)*4)))
            writer.writerow({"epoch": i+1, "train_loss": float(train_loss/(dt+1)),
                             "val_loss": float(val_loss/((j+1)*4))})

    return model


def test_SEM(model, data_test,  folder_to_save):
    """Test the model with test dataset
    Args:
        model: model to be tested
        data_test (DataLoader): test dataset
        folder_to_save (str): path that the predictions would be saved
    """
    for i, (images) in enumerate(data_test):

        print(i)

        print(images)
        stacked_img = torch.Tensor([])
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0).cuda())
            output = model(image.cuda())
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
