import torch.nn.functional as F
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


# class BBox_Predictor(nn.Module):
#     """ BBox Predictor module"""
#     def __init__(self, inplanes=64):
#         super(BBox_Predictor, self).__init__()
#         self.conv1 = conv(inplanes, inplanes, kernel_size=1, padding=0)
#         self.conv2 = conv(inplanes, inplanes, kernel_size=1, padding=0)
#         self.avgpool = nn.Sequential(
#             nn.Conv2d(inplanes, inplanes, kernel_size=(16, 16), padding=False),
#             nn.BatchNorm2d(inplanes),
#             nn.ReLU(),
#         )
#         # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Sequential(nn.Linear(inplanes, 4),nn.ReLU())
#         self.fc = nn.Sequential(nn.Linear(inplanes, 4))
#
#     def forward(self, x):
#         """ Forward pass with input x. """
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


class BBox_Predictor(nn.Module):
    """ BBox Predictor module"""
    def __init__(self, inplanes=64):
        super(BBox_Predictor, self).__init__()
        self.conv1 = conv(inplanes, inplanes*8, kernel_size=3, padding=1)
        self.conv2 = conv(inplanes*8, inplanes*4, kernel_size=3, padding=1)
        self.conv3 = conv(inplanes*4, inplanes*2, kernel_size=3, padding=1)
        self.conv4 = conv(inplanes*2, inplanes, kernel_size=3, padding=1)
        self.avgpool = nn.Sequential(
            nn.Conv2d(inplanes, inplanes*4, kernel_size=(16, 16), padding=False),
            nn.BatchNorm2d(inplanes*4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(inplanes*4, 4))

    def forward(self, x):
        """ Forward pass with input x. """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BBox_Predictor2(nn.Module):
    """ BBox Predictor module"""
    def __init__(self, inplanes=64):
        super(BBox_Predictor2, self).__init__()
        self.conv1 = conv(inplanes, inplanes*2, kernel_size=3, padding=1)
        self.conv2 = conv(inplanes*2, inplanes*2, kernel_size=3, padding=1)
        self.conv3 = conv(inplanes*2, inplanes*2, kernel_size=3, padding=1)
        self.conv4 = conv(inplanes*2, inplanes*2, kernel_size=3, padding=1)
        self.avgpool = nn.Sequential(
            nn.Conv2d(inplanes*2, inplanes*4, kernel_size=(16, 16), padding=False),
            nn.BatchNorm2d(inplanes*4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(inplanes*4, 4))

    def forward(self, x):
        """ Forward pass with input x. """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
