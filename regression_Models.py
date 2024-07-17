
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


def resnet_all(in_channels=13, resnetX='resnet18'):
    model = timm.create_model(f'{resnetX}')

    model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, 
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model