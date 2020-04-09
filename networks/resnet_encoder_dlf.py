from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def dynamic_local_filtering(x, depth, dilated=1):
    padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
    pad_depth = padding(depth)
    n, c, h, w = x.size()
    # y = torch.cat((x[:, int(c/2):, :, :], x[:, :int(c/2), :, :]), dim=1)
    # x = x + y
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
    return filter / 9

class ResNetMultiImageInput(modes.Resnet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, using_dlf=False, using_gn=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images*3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if using_gn:
            self.normal = nn.GroupNorm(16,64)
        else:
            self.normal = nn.BatchNorm2d(64)
        self.elu = nn.elu()

        

