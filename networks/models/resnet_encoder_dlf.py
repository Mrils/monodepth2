from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from networks.models import resnet_v2

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

class ResNet_DLF(modes.Resnet):
    def __init__(self,base_model=18,base_seg_model=18, nums_input_images=1):
        super(ResNet_DLF, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if base_model > 34:
            self.num_ch_enc[1:] *= 4
        self.base = resnet_v2.ResnetBase(base_model, nums_input_images=nums_input_images)
        self.segnet = resnet_v2.ResnetBase(base_seg_model, nums_input_images=nums_input_images)

        if self.adaptive_diated:
            self.adaptive_softmax = nn.Softmax(dim=3)

            self.adaptive_layers = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(512, 512 * 3, 3, padding=0),
            )
            self.adaptive_bn = nn.BatchNorm2d(512)
            self.adaptive_relu = nn.ReLU(inplace=True)

            self.adaptive_layers1 = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(1024, 1024 * 3, 3, padding=0),
            )
            self.adaptive_bn1 = nn.BatchNorm2d(1024)
            self.adaptive_relu1 = nn.ReLU(inplace=True)

        def forward(self,x,seg):
            batch_size = x.size(0)

            x = self.base.conv1(x)
            depth = self.depthnet.conv1(depth)
            x = self.base.bn1(x)
            depth = self.depthnet.bn1(depth)
            x = self.base.relu(x)
            depth = self.depthnet.relu(depth)
            x = self.base.maxpool(x)
            depth = self.depthnet.maxpool(depth)

            x = self.base.layer1(x)
            depth = self.depthnet.layer1(depth)
            # x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

            x = self.base.layer2(x)
            depth = self.depthnet.layer2(depth)

            if self.adaptive_diated:
                weight = self.adaptive_layers(x).reshape(-1, 512, 1, 3)
                weight = self.adaptive_softmax(weight)
                x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                    + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                    + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
                x = self.adaptive_bn(x)
                x = self.adaptive_relu(x)
            else:
                x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

            # if self.use_dropout and self.dropout_position == 'adaptive':
            #     x = self.dropout(x)

            # if self.drop_channel:
            #     x = self.dropout_channel(x)

            x = self.base.layer3(x)
            depth = self.depthnet.layer3(depth)

            if self.adaptive_diated:
                weight = self.adaptive_layers1(x).reshape(-1, 1024, 1, 3)
                weight = self.adaptive_softmax(weight)
                x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                    + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                    + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
                x = self.adaptive_bn1(x)
                x = self.adaptive_relu1(x)
            else:
                x = x * depth

            x = self.base.layer4(x)
            depth = self.depthnet.layer4(depth)
            x = x * depth



        

