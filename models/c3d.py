# 2020.10.07 add C3D model

import torch
from net_part import *

class C3D(nn.Module):

    def __init__(self, with_classifier=False, num_classes=101, return_feature=False):
        super(C3D, self).__init__()
        self.with_classifier = with_classifier
        self.return_feature = return_feature
        self.num_classes = num_classes

        self.conv1 = conv3d(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv3d(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3_1 = conv3d(128, 256)
        self.conv3_2 = conv3d(256, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4_1 = conv3d(256, 512)
        self.conv4_2 = conv3d(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5_1 = conv3d(512, 512)
        self.conv5_2 = conv3d(512, 512)
        
        # return feature map
        if self.return_feature:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # classifier
        if self.with_classifier:
            self.pool5 = nn.AdaptiveAvgPool3d(1)
            self.linear = nn.Linear(512, self.num_classes)

    
    def subforward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        # get feature map
        return x

    def forward(self, x):
        x = self.subforward(x)

        if self.return_feature == True:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)

        if self.with_classifier == True:
            x = self.pool5(x)
            x = x.view(-1, x.size(1))
            x = self.linear(x)

        return x

if __name__ == '__main__':
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 112, 112))
    c3d = C3D(with_classifier=False, num_classes=101, return_feature=True)
    output = c3d(input_tensor)

    print(output.shape)