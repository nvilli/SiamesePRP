# 2020.10.07 Siamese 3D network(one C3D network as a single stack)

import torch
import torch.nn as nn
from c3d import C3D
from net_part import *

class Siamese3D(nn.Module):

    def __init__(self, base_network, with_classifier=False, adaptive_contact=False, num_classes=4):
        super(Siamese3D, self).__init__()

        self.base_network = base_network
        self.with_classifier = with_classifier
        self.adaptive_contact = adaptive_contact
        self.num_classes = num_classes

        if self.with_classifier == True:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x_original, x_sample, tuple_order=None):

        x_original = self.base_network(x_original)
        x_sample   = self.base_network(x_sample)

        if self.return_feature == True:
            x_original = self.pool(x_original)
            x1 = x_original.view(-1, x_original.size[1])
            x1 = self.fc(x1)

            x_sample = self.pool(x_sample)
            x2 = x_sample.view(-1, x_sample.size[1])
            x2 = self.fc(x2)

        
        if self.with_classifier:
            return x1, x2