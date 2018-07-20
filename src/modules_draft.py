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

def train_CE_SEM(model, criterion, epoch, img_folder, mask_folder, num_workers=4, batch_size=1, learning_rate=0.01):
    SEM_train = SEMDataTrain(img_folder, mask_folder, in_size=572, out_size=388)
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train, num_workers=num_workers,
                                    batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("initializing training!")
    with open('train_loss.csv', 'w') as csvfile:
        fieldnames = ['epoch', 'image' ,'loss']
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
                writer.writerow({'epoch' : i+1 , 'image' : j+1, 'loss' : float(loss)})
                break
    return model


def test_SEM(model, loss_function, epoch, img_folder, mask_folder, folder_to_save):
    SEM_test = SEMDataTest(img_folder, mask_folder, in_size=572, out_size=388)
    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test, num_workers=4,
                                    batch_size=1, shuffle=True)
    print(SEM_test_load)
    criterion = loss_function

    for single_epoch in range(epoch):
        for i, (images) in enumerate(SEM_test_load):
            stacked_img = torch.Tensor([])
            for j in range(images.size()[1]):
                image = Variable(images[:, j, :, :].unsqueeze(0))
                output = model(image)
                print('output', output)
                #output = softmax(output, dim=1)
                #print("size", output.size())
                output = torch.argmax(output, dim=1).float()
                #print("size", output.size())
                stacked_img = torch.cat((stacked_img, output))

            div_arr = division_array(388, 2, 2, 512, 512)

            img_cont = image_concatenate(polarize(stacked_img.data.numpy()), 2, 2, 512, 512)
            final_img = (img_cont*255/div_arr)

            final_img = final_img.astype("uint8")

            # Save output images after 100 epochs
            # also, ignore every 2 epoch to see more noticeable changes.
            if single_epoch % 1 == 0 and single_epoch >= 1:

                final_img = Image.fromarray(final_img)
                # organize images in every epoch
                desired_path = 'result_images/epoch_' + str(single_epoch) + '/'

                # Create the path if it does not exist
                if not os.path.exists(desired_path):
                    os.makedirs(desired_path)

                # Save Image!
                export_name = 'test' + str(i) + '.png'
                final_img.save(desired_path + export_name)
            if i == 3:
                break
    return final_img


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img
