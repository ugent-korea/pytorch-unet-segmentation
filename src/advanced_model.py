# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


class Double_conv(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_down(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


class Conv_up(nn.Module):

    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2),
                          int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class CleanU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CleanU_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)
        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)
        #self.Conv_final = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)

    def forward(self, x):

        x, conv1 = self.Conv_down1(x)
        #print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        #print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        #print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        #print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        #print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        #print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        #print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        #print("up3 => uConv3|", x.shape)
        x = self.Conv_up4(x, conv1)
        x = self.Conv_out(x)
        #x = self.Conv_final(x)

        return x


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 572, 572)
    model = CleanU_Net(1, 2)
    x = model(im)
    print(x.shape)
    del model
    del x
    # print(x.shape)
