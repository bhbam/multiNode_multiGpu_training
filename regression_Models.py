
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

import torch.nn as nn
import torch.hub
import torch.nn.functional as F
import timm
# ---------More deeply Modified old resnet without ieta and iphi-------------------------------------------------------------------------------------------------

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks


        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=1, padding=1)

        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])

        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])

        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])

        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])

        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])

        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])

        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])

        self.fc = nn.Linear(self.fmaps[3], 1)
        self.GlobalMaxPool2d = nn.AdaptiveMaxPool2d((1,1))

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, X):

        x = self.conv0(X)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.GlobalMaxPool2d(x)
        x = x.view(x.size()[0], self.fmaps[3])
        x = self.fc(x)
        return x
# model = ResNet(in_channels=len(indices), nblocks=3, fmaps=[8,16,32,64])
#-------------------------------------------------------------------------------------

#create efficientNet model
def EfficientNet(in_channels=13, effnet=0 ):

    # Create EfficientNet model
    model = timm.create_model(f"efficientnet_b{effnet}", num_classes=1)

    # Modify the conv_stem layer
    model.conv_stem = nn.Conv2d(in_channels,
                                model.conv_stem.out_channels,
                                model.conv_stem.kernel_size,
                                model.conv_stem.stride,
                                model.conv_stem.padding,
                                bias=False)

    return model

# model = EfficientNet(in_channels=len(indices), effnet=0)
   
#-------------------------------------------------------------------------------------
#create resnet model
def resnet_all(in_channels=13, resnetX='resnet18'):
    model = timm.create_model(f'{resnetX}')

    model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, 
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
# model = resnet_all(in_channels=len(indices), resnetX='resnet18')

#-------------------------------------------------------------------------------------

# Create  CoAtNet model
class CustomCoAtNet(nn.Module):
    def __init__(self, in_channels=13, coatnet='coatnet_0_224'):
        super(CustomCoAtNet, self).__init__()
        self.model = timm.create_model(f'{coatnet}')
        
        # Modify the stem layer to accept `in_channels`
        self.model.stem.conv1 = nn.Conv2d(in_channels, self.model.stem.conv1.out_channels, kernel_size=self.model.stem.conv1.kernel_size,
                                          stride=self.model.stem.conv1.stride, padding=self.model.stem.conv1.padding, bias=False)
        
        # Modify the final layer for regression
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, 1)

    def forward(self, x):
        # Resize the input tensor to (224, 224) within the forward method
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return self.model(x)

# model = CustomCoAtNet(in_channels=13, coatnet='coatnet_0_224')

#-------------------------------------------------------------------------------------