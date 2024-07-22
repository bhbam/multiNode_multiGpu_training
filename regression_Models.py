
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