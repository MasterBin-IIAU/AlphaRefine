import torch.nn as nn
import torch
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256):
        super(Predictor, self).__init__()

        '''top-left corner'''
        self.conv1 = conv(inplanes, channel)
        self.conv2 = conv(channel, channel)
        self.conv3 = conv(channel, channel)
        self.conv4 = conv(channel, channel*2)
        self.conv5 = nn.Conv2d(2*channel, 8, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.output_sz = 16  # same as input resolution
            self.indice = torch.arange(0, self.output_sz).view(-1, 1) * 16
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.output_sz, 1)) \
                .view((self.output_sz * self.output_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.output_sz)) \
                .view((self.output_sz * self.output_sz,)).float().cuda()

    def forward(self, x):
        """ Forward pass with input x. """
        indication_map, offset_map = self.get_score_map(x)
        xyxy = self.soft_argmax(indication_map, offset_map)
        return xyxy

    def get_score_map(self, x):

        #top-left branch
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        raw = self.conv5(x4)

        indication_map = raw[:, :4, :, :]
        offset_map = raw[:, 4:, :, :]

        return indication_map, offset_map

    def get_heatmap(self, x):
        score_map_tl, score_map_br = self.get_score_map(x)
        heatmap_tl = self.score2heat(score_map_tl)
        heatmap_br = self.score2heat(score_map_br)
        return heatmap_tl, heatmap_br

    def score2heat(self, score_map):
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.output_sz * self.output_sz)),dim=1)  # (batch, output_sz * output_sz)
        heatmap = prob_vec.view((-1, self.output_sz, self.output_sz))
        return heatmap

    def soft_argmax(self, indication_map, offset_map):
        """ get soft-argmax coordinate for a given heatmap """
        offset_map = offset_map.reshape(*offset_map.shape[:2], -1)\
                     + torch.stack([self.coord_x, self.coord_y, self.coord_x, self.coord_y], dim=0)
        prob_vec = nn.functional.softmax(
            indication_map.view((*indication_map.shape[:2], self.output_sz * self.output_sz)), dim=-1)  # (batch, output_sz * output_sz)
        xyxy = torch.sum((offset_map * prob_vec), dim=-1)
        return xyxy


if __name__ == '__main__':
    indication_map = torch.ones((1, 4, 16, 16)) * (-5)
    offset_map = torch.ones((1, 4, 16, 16))
    indication_map[0, 0, 11, 2] = 20.0
    indication_map[0, 1, 12, 2] = 20.0
    indication_map[0, 2, 13, 2] = 20.0
    indication_map[0, 3, 14, 2] = 20.0
    offset_map[0, :, 11, 2] = 0.1
    offset_map[0, :, 12, 2] = 0.2
    offset_map[0, :, 13, 2] = 0.3
    offset_map[0, :, 14, 2] = 0.4
    indication_map = indication_map.cuda()
    offset_map = offset_map.cuda()
    corner_predictor = Predictor()

    xyxy = corner_predictor.soft_argmax(indication_map, offset_map)
    print(xyxy)
