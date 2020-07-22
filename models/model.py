import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.metrics import AdaCos
#from models.osnet import osnet_ibn_x1_0
from models.osnet import osnet_x1_0
from models.metrics import ArcMarginProduct
import pytorch_pfn_extras as ppe
import numpy as np


class Model(nn.Module):
    def __init__(self, n_classes, feature_extractor=None, metric_learning=False, emb_dims=512):
        super().__init__()
        self.ef = True if 'efficient' in feature_extractor else False
        self.ef_rb = True if 'rfb_ef' in feature_extractor else False
        self.ef_as = True if 'as_ef' in feature_extractor else False
        self.osnet = True if 'osnet' in feature_extractor else False
        self.metric_learning = metric_learning
        ef_last_ch = {'efficientnet-b0':1280, 'efficientnet-b1':1280,'efficientnet-b2':1408,
                      'efficientnet-b3':1536, 'efficientnet-b4':1792, 'efficientnet-b5':2048}
        if self.ef:
            from efficientnet_pytorch import EfficientNet
            self.features = EfficientNet.from_pretrained(feature_extractor)
            #last_ch = 1280
            last_ch = ef_last_ch[feature_extractor]
        elif self.osnet:
            self.features = osnet_x1_0(pretrained=True)
            last_ch = 1000
        elif self.ef_rb:
            feature_extractor = 'efficientnet-{}'.format(feature_extractor.split('-')[-1])
            from models.efficientnet import EfficientNetRFB
            #self.features = EfficientNetRFB(feature_extractor, output_blocks_ind=[2, 4, 10, 15])
            self.features = EfficientNetRFB(feature_extractor, output_blocks_ind=[5, 9, 15,21,26,31])
            last_ch = ef_last_ch[feature_extractor]
        elif self.ef_as:
            feature_extractor = 'efficientnet-{}'.format(feature_extractor.split('-')[-1])
            from models.aspp import ASPP
            from models.efficientnet import EfficientNetASPP
            last_ch = ef_last_ch[feature_extractor]
            aspp = ASPP(output_stride=32, BatchNorm=nn.BatchNorm2d, inplanes=last_ch)
            self.features = EfficientNetASPP(feature_extractor, aspp)
        else:
            self.features = models.resnet34(pretrained=True)
            last_ch = 1000
        if not metric_learning:
            fc_ch = 1280
            if last_ch == 2048:
                fc_ch = 1680
            self.fc0 = nn.Linear(last_ch, fc_ch)
            self.fc1 = nn.Linear(fc_ch, n_classes)
        else:
            self.bn0 = nn.BatchNorm2d(last_ch)
            self.drop_out = nn.Dropout()
            self.fc = nn.Linear(last_ch * 16 * 16, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.emb = AdaCos(512, 4)
            #self.emb = ArcMarginProduct(512, 4)
        self.last_ch = last_ch

    def forward(self, x, t=None):
        if self.ef or self.ef_rb or self.ef_as:
            if self.ef:
                h = self.features.extract_features(x)
            else:
                h = self.features(x)
            if not self.metric_learning:
                h = F.avg_pool2d(h, h.size()[2:]).reshape((-1, self.last_ch))
        else:
            h = self.features(x)

        if not self.metric_learning:
            h = self.fc0(h)
            h = self.fc1(h)
        else:
            h = self.bn0(h)
            h = self.drop_out(h)
            h = h.view(h.size(0), -1)
            h = self.fc(h)
            h = self.bn1(h)
            #if t is not None:
            h = self.emb(h, t)
        return h