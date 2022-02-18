import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision


class SingleStream(nn.Module):
    """ Constructs Resnet-Deeplabv3 segmentation model with additional streams for depth inputs
    """
    def __init__(self, num_classes=4, pretrained=True, use_rgb=True, use_depth=False):
        """
        @param num_classes (int): Number of classes including background
        @param pretrained (bool): Whether to load Coco pretrained weights into Resnet backbone
        @param use_depth (bool): Whether to increase number of input channels to allow for depth
        """
        super().__init__()
        self.use_depth = use_depth
        self.use_rgb = use_rgb
        
        if self.use_depth and self.use_rgb:
            num_channels = 8 # Training with both rgb and depth
        elif not self.use_depth and self.use_rgb:
            num_channels = 6 # Training with only rgb
        elif self.use_depth and not self.use_rgb:
            num_channels = 2 # Training with only depth
        else:
            print("Need RGB and/or depth input params set.")
            raise Exception

        self.model = torchvision.models.segmentation.segmentation.deeplabv3_resnet50(pretrained=False)
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.model.backbone.load_state_dict(state_dict)
            print("Loaded weights into backbone")

        self.model.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.conv1_relu = nn.ReLU()
        self.conv1_norm = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        img1, img2, depth1, depth2 = x

        if self.use_depth and self.use_rgb:
            input_cat = torch.cat((img1, depth1, img2, depth2), dim=1)
        elif not self.use_depth and self.use_rgb:
            input_cat = torch.cat((img1, img2), dim=1)
        elif self.use_depth and not self.use_rgb:
            input_cat = torch.cat((depth1, depth2), dim=1)
        else:
            print("Need RGB and/or depth input.")
            raise Exception

        x = self.conv1(input_cat)
        x = self.conv1_norm(self.conv1_relu(x))

        x = self.model(x)

        return x["out"]


