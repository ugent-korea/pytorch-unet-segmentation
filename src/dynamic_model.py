# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
from model_modules import *


class CleanU_Net(nn.Module):
    def __init__(self, in_channels):
        super(CleanU_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 32)
        self.Conv_down2 = Conv_down(32, 64)
        self.Conv_down3 = Conv_down(64, 128)
        self.Conv_down4 = Conv_down(128, 256)
        self.Conv_down5 = Conv_down(256, 512)
        self.Conv_up1 = Conv_up(512, 256)
        self.Conv_up2 = Conv_up(256, 128)
        self.Conv_up3 = Conv_up(128, 64)
        self.Conv_up4 = Conv_up(64, 32)

    def forward(self, x):

        x, conv1 = self.Conv_down1(x)
        print(x.shape)
        x, conv2 = self.Conv_down2(x)
        print(x.shape)
        x, conv3 = self.Conv_down3(x)
        print(x.shape)
        x, conv4 = self.Conv_down4(x)
        print(x.shape)
        _, x = self.Conv_down5(x)
        print(x.shape)
        x = self.Conv_up1(x, conv4)
        print(x.shape)
        x = self.Conv_up2(x, conv3)
        print(x.shape)
        x = self.Conv_up3(x, conv2)
        print(x.shape)
        x = self.Conv_up4(x, conv1)
        return x


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 572, 572)
    model = CleanU_Net(1)
    x = model(im)
    print(x.shape)
    del model
    del x
    # print(x.shape)
