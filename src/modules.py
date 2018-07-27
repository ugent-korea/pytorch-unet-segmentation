import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import *
import torch.nn as nn
from accuracy import accuracy_check, accuracy_check_for_batch
import csv
import os


def train_model(model, data_train, criterion, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train()
    for batch, (images, masks) in enumerate(data_train):
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)
        # print(masks.shape, outputs.shape)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
    # total_loss = get_loss_train(model, data_train, criterion)


def get_loss_train(model, data_train, criterion):
    """
        Calculate loss over train set
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad():
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc/(batch+1), total_loss/(batch + 1)


def validate_model(model, data_val, criterion, epoch, make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    total_val_loss = 0
    total_val_acc = 0
    for batch, (images_v, masks_v, original_msk) in enumerate(data_val):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                mask_v = Variable(masks_v[:, index, :, :].squeeze(1).cuda())
                # print(image_v.shape, mask_v.shape)
                output_v = model(image_v)
                total_val_loss = total_val_loss + criterion(output_v, mask_v).cpu().item()
                # print('out', output_v.shape)
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))
        if make_prediction:
            im_name = batch  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)
            acc_val = accuracy_check(original_msk, pred_msk)
            total_val_acc = total_val_acc + acc_val

    return total_val_acc/(batch + 1), total_val_loss/((batch + 1)*4)


def test_model(model_path, data_test, epoch, save_folder_name='prediction'):
    """
        Test run
    """
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    model.eval()
    for batch, (images_t) in enumerate(data_test):
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_t.size()[1]):
            with torch.no_grad():
                image_t = Variable(images_t[:, index, :, :].unsqueeze(0).cuda())
                # print(image_v.shape, mask_v.shape)
                output_t = model(image_t)
                output_t = torch.argmax(output_t, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_t))
        im_name = batch  # TODO: Change this to real image name so we know
        _ = save_prediction_image(stacked_img, im_name, epoch, save_folder_name)
    print("Finish Prediction!")


def save_prediction_image(stacked_img, im_name, epoch, save_folder_name="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """
    div_arr = division_array(388, 2, 2, 512, 512)
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), 2, 2, 512, 512)
    img_cont = polarize((img_cont)/div_arr)*255
    img_cont_np = img_cont.astype('uint8')
    img_cont = Image.fromarray(img_cont_np)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img_cont.save(desired_path + export_name)
    return img_cont_np


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


"""
def test_SEM(model, data_test,  folder_to_save):
    '''Test the model with test dataset
    Args:
        model: model to be tested
        data_test (DataLoader): test dataset
        folder_to_save (str): path that the predictions would be saved
    '''
    for i, (images) in enumerate(data_test):

        print(images)
        stacked_img = torch.Tensor([])
        for j in range(images.size()[1]):
            image = Variable(images[:, j, :, :].unsqueeze(0).cuda())
            output = model(image.cuda())
            print(output)
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
"""


if __name__ == '__main__':
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train,
                                                 num_workers=3, batch_size=10, shuffle=True)
    get_loss_train()
