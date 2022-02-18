import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision


class TwoStream(nn.Module):
    """ Constructs semantic differencing model with a shared Resnet50 encoder and two heads, 
    one for segmentation and one for monocular depth estimation.
    """
    def __init__(self, num_classes=4, pretrained=True):
        """
        @param num_classes (int): Number of classes including background
        @param pretrained (bool): Whether to load Coco pretrained weights into Resnet backbone
        """
        super().__init__()
        # Encoder params
        num_channels = 8

        self.encoder = torchvision.models.segmentation.segmentation.resnet50(pretrained=False)
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.encoder.backbone.load_state_dict(state_dict)
            print("Loaded weights into backbone")

        self.encoder.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.conv1_relu = nn.ReLU()
        self.conv1_norm = nn.BatchNorm2d(num_channels)

        # Segmentation head params
        self.seg_head = DeeplabV3Head()

        # Monocular depth estimation head params
        self.depth_head = 

    def forward(self, x):
        img1, img2, depth1, depth2 = x

        input_cat = torch.cat((img1, depth1, img2, depth2), dim=1)

        x = self.conv1(input_cat)
        x = self.conv1_norm(self.conv1_relu(x))

        enc_out = self.encoder(x)
        results = OrderedDict()
        results["segmentation"] = self.seg_head(enc_out)
        results["depth"] = self.depth_head(enc_out)

        return results


