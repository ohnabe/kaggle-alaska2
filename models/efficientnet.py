"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

from efficientnet_pytorch import EfficientNet as EfficientNetBase
from torch import nn
import torch
from models.rfb import BasicRFB_c
import numpy as np


class EfficientNetRFB(nn.Module):
    def __init__(self, feature_extractor, num_classes=4, output_blocks_ind=None):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(feature_extractor)
        self.efficient_net._set_output_blocks_ind(output_blocks_ind)
        self.efficient_net.get_feature_map_shape(torch.tensor(np.ones((1, 3, 512, 512), dtype=np.float32)))

        with torch.no_grad():
            self.efficient_net.get_feature_map_shape(torch.tensor(np.ones((1, 3, 512, 512), dtype=np.float32)))

        feature_map_size = self.efficient_net.feature_map_size

        layers = []


        for i in range(len(output_blocks_ind)):
            ind = output_blocks_ind[i]
            if i == 0:
                in_ch = feature_map_size[ind][1]
            else:
                pre_ind = output_blocks_ind[i-1]
                in_ch = feature_map_size[pre_ind][1] + feature_map_size[ind][1]
            out_ch = feature_map_size[ind][1]
            layers.append(BasicRFB_c(in_ch, out_ch))

        self.rfb = nn.ModuleList(layers)
        self.last_conv = nn.Conv2d(feature_map_size[-2][1]+feature_map_size[-1][1], feature_map_size[-1][1], 1, bias=False)
        self.last_bn = nn.BatchNorm2d(feature_map_size[-1][1])

        #self.fc0 = nn.Linear(feature_map_size[-1][1], 1280)
        #self.fc1 = nn.Linear(1280, num_classes)

    def forward(self, x):
        h, ys = self.efficient_net.extract_features(x)
        #print(len(ys))
        #print(len(self.rfb))

        for i, rfb in enumerate(self.rfb):
            if i == 0:
                r = rfb(ys[i])
            else:
                if ys[i].size()[2:] != r.size()[2:]:
                    r = nn.functional.max_pool2d(r, kernel_size=2, stride=2)
                r = rfb(torch.cat((ys[i], r), 1))

        if h.size()[2:] != r.size()[2:]:
            r = nn.functional.max_pool2d(r, kernel_size=2, stride=2)

        h = self.efficient_net._swish(self.last_bn(self.last_conv(torch.cat((h, r), 1))))
        #h = nn.functional.avg_pool2d(h, h.size[2:]).view(h.size(0), -1)
        #h = self.fc0(h)
        #h = self.fc1(h)
        return h


class EfficientNetASPP(nn.Module):
    def __init__(self, feature_extractor, aspp, num_classes=4):
        super(EfficientNetASPP, self).__init__()
        self.efficient_net = EfficientNetBase.from_pretrained(feature_extractor)
        self.aspp = aspp

    def forward(self, x):
        h = self.efficient_net.extract_features(x)
        h = self.aspp(h)
        return h



class EfficientNet(EfficientNetBase):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__(blocks_args, global_params)

    def _set_output_blocks_ind(self, output_blocks_ind):
        self.output_blocks_ind = output_blocks_ind

    def extract_features(self , inputs):
        ys = []
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if len(self.output_blocks_ind) > 0:
                if idx == self.output_blocks_ind[0]:
                    ys.append(x)
                    self.output_blocks_ind.append(self.output_blocks_ind[0])
                    self.output_blocks_ind.pop(0)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x, ys


    def get_feature_map_shape(self, inputs):
        ys = []
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            ys.append(x.size())
        x = self._swish(self._bn1(self._conv_head(x)))
        ys.append(x.size())

        self.feature_map_size = ys
        print(ys)


if __name__ == '__main__':
    ef = EfficientNet.from_pretrained('efficientnet-b0')

    import numpy as np
    with torch.no_grad():
        ys = ef.get_feature_map_shape(torch.tensor(np.ones((1, 3, 512, 512), dtype=np.float32)))
        print(ys)