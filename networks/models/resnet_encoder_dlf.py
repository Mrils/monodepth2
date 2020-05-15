from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from networks.models import resnet_v2
from networks.models import selayer

def dynamic_local_filtering(x, seg, dilated=1):
    padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
    pad_seg = padding(seg)
    n, c, h, w = x.size()
    # y = torch.cat((x[:, int(c/2):, :, :], x[:, :int(c/2), :, :]), dim=1)
    # x = x + y
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (pad_seg[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (pad_seg[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
    return filter / 9

class ResNet_DLF(nn.Module):

    def __init__(self,base_model=18,base_seg_model=18, nums_input_images=1, adaptive_diated=True):
        super(ResNet_DLF, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        if base_model > 34:
            self.num_ch_enc[1:] *= 4
        self.base = resnet_v2.ResnetBase(base_model, num_input_images=nums_input_images)
        self.segnet = resnet_v2.ResnetBase(base_seg_model, num_input_images=nums_input_images)
        self.adaptive_diated = adaptive_diated

        if self.adaptive_diated:
            self.adaptive_softmax = nn.Softmax(dim=3)

            self.adaptive_layers = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(self.num_ch_enc[2], self.num_ch_enc[2] * 3, 3, padding=0),
            )
            self.adaptive_bn = nn.BatchNorm2d(self.num_ch_enc[2])
            self.adaptive_relu = nn.ReLU(inplace=True)

            self.adaptive_layers1 = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(self.num_ch_enc[3], self.num_ch_enc[3] * 3, 3, padding=0),
            )
            self.adaptive_bn1 = nn.BatchNorm2d(self.num_ch_enc[3])
            self.adaptive_relu1 = nn.ReLU(inplace=True)

    def forward(self,x,seg):
        batch_size = x.size(0)

        features=[]
        x = (x - 0.45) / 0.225
        x = self.base.conv1(x)
        seg = self.segnet.conv1(seg)
        x = self.base.bn1(x)
        seg = self.segnet.bn1(seg)
        x = self.base.relu(x)
        features.append(x)

        seg = self.segnet.relu(seg)
        x = self.base.maxpool(x)
        seg = self.segnet.maxpool(seg)

        x = self.base.layer["layer_1"](x)
        features.append(x)

        seg = self.segnet.layer["layer_1"](seg)
        # x = dynamic_local_filtering(x, seg, dilated=1) + dynamic_local_filtering(x, seg, dilated=2) + dynamic_local_filtering(x, seg, dilated=3)

        x = self.base.layer["layer_2"](x)
        features.append(x)

        seg = self.segnet.layer["layer_2"](seg)

        if self.adaptive_diated:
            weight = self.adaptive_layers(x).reshape(-1, x.size()[1], 1, 3)
            weight = self.adaptive_softmax(weight)
            x = dynamic_local_filtering(x, seg, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, seg, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, seg, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn(x)
            x = self.adaptive_relu(x)
        else:
            x = dynamic_local_filtering(x, seg, dilated=1) + dynamic_local_filtering(x, seg, dilated=2) + dynamic_local_filtering(x, seg, dilated=3)
        # if self.use_dropout and self.dropout_position == 'adaptive':
        #     x = self.dropout(x)

        # if self.drop_channel:
        #     x = self.dropout_channel(x)

        x = self.base.layer["layer_3"](x)
        seg = self.segnet.layer["layer_3"](seg)

        if self.adaptive_diated:
            weight = self.adaptive_layers1(x).reshape(-1, x.size()[1], 1, 3)
            weight = self.adaptive_softmax(weight)
            x = dynamic_local_filtering(x, seg, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, seg, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, seg, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn1(x)
            x = self.adaptive_relu1(x)
        else:
            x = x * seg
        features.append(x)
        x = self.base.layer["layer_4"](x)
        seg = self.segnet.layer["layer_4"](seg)
        x = x * seg
        features.append(x)


        return features

class ResNet_SE(nn.Module):
    def __init__(self,base_model=18,base_seg_model=18, nums_input_images=1, using_se=True, se_layers=[4,3,2,1]):
        super(ResNet_SE, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        if base_model > 34:
            self.num_ch_enc[1:] *= 4
        self.base = resnet_v2.ResnetBase(base_model, num_input_images=nums_input_images)
        self.segnet = resnet_v2.ResnetBase(base_seg_model, num_input_images=nums_input_images)
        self.using_se = using_se
        self.se_layers = se_layers
        
        if self.using_se:
            self.se = nn.ModuleDict()
            self.shrink = nn.ModuleDict()
            for index, layer in enumerate(self.num_ch_enc):
                if index in self.se_layers:
                    self.se["se_{}".format(index)] = selayer.SElayer(layer*2)
                    self.shrink["shrink_{}".format(index)] = nn.Conv2d(layer*2,layer,1)


    def forward(self,x,seg):
        batch_size = x.size(0)

        features=[]
        x = (x - 0.45) / 0.225
        seg = (seg - 0.45) / 0.225
        x = self.base.conv1(x)
        seg = self.segnet.conv1(seg)
        x = self.base.bn1(x)
        seg = self.segnet.bn1(seg)
        x = self.base.relu(x)
        seg = self.segnet.relu(seg)

        features.append(x)

        x = self.base.maxpool(x)
        seg = self.segnet.maxpool(seg)
        

        for i in range(1,5):
            x = self.base.layer["layer_{}".format(i)](x)
            seg = self.segnet.layer["layer_{}".format(i)](seg)
            if self.using_se and i in self.se_layers:
                x = torch.cat((x,seg),dim=1)
                x = self.se["se_{}".format(i)](x)
                x = self.shrink["shrink_{}".format(i)](x)
                # print(x.shape)
            features.append(x)

        return features


        

