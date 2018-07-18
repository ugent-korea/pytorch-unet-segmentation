from dynamic_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
from torch.utils.data.dataset import Dataset

if __name__ == "__main__":
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')

    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=8, batch_size=1, shuffle=True)
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=8, batch_size=2, shuffle=True)

    model = CleanU_Net(in_channels=1)

    print("TRAIN DATA")
    for i, (images, labels) in enumerate(SEM_train_load):
        images = Variable(images)
        labels = Variable(labels)
        print("input size|", images.size())
        output = model(images)
        print("output size|", output.shape)
        break

    print("\n")

    print("TEST DATA")
    for i, (images) in enumerate(SEM_test_load):
        for j in range(images.size()[1]):
            print(images.size())
            image = Variable(images[:, j, :, :].unsqueeze(0))
            print("input size|", image.size())
            output = model(image)
            print("output size|", output.shape)
            break
        break

    del model
    del output
