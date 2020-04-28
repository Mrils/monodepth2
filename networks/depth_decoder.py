# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from networks.models import selayer

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec, scales=range(4), num_output_channels=1, use_skips=True, has_se=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.has_se = has_se

        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_dec = num_ch_dec

        # decoder
        self.convs = nn.ModuleDict()
        # self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv_{}_0".format(i)] = ConvBlock(num_ch_in, num_ch_out)
            # self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
                # num_ch_in *= 2
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv_{}_1".format(i)] = ConvBlock(num_ch_in, num_ch_out)
            if self.has_se:
                self.convs["se_{}".format(i)] = selayer.SElayer(num_ch_in)

        for s in self.scales:
            self.convs["dispconv_{}".format(s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs["upconv_{}_0".format(i)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if self.has_se:
                x = self.convs["se_{}".format(i)](x)
            x = self.convs["upconv_{}_1".format(i)](x)
           
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs["dispconv_{}".format(i)](x))

        return self.outputs
