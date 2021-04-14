import torch.nn as nn
import torch
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class RPNHead(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256):
        super(RPNHead, self).__init__()

        '''top-left corner'''
        self.conv1_box = conv(inplanes, channel)
        self.conv2_box = conv(channel, channel)
        self.conv3_box = conv(channel, channel)
        self.conv4_box = conv(channel, channel)
        self.conv5_box = nn.Conv2d(channel, 4, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_conf = conv(inplanes, channel)
        self.conv2_conf = conv(channel, channel)
        self.conv3_conf = conv(channel, channel)
        self.conv4_conf = conv(channel, channel)
        self.conv5_conf = nn.Conv2d(channel, 1, kernel_size=1)

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
        boxes, confs = self.predict(x)
        box = self.get_box(boxes, confs)
        return box

    def predict(self, x):
        x_box1 = self.conv1_box(x)
        x_box2 = self.conv2_box(x_box1)
        x_box3 = self.conv3_box(x_box2)
        x_box4 = self.conv4_box(x_box3)
        boxes = self.conv5_box(x_box4)

        x_conf1 = self.conv1_conf(x)
        x_conf2 = self.conv2_conf(x_conf1)
        x_conf3 = self.conv3_conf(x_conf2)
        x_conf4 = self.conv4_conf(x_conf3)
        confs = self.conv5_conf(x_conf4)

        return boxes, confs

    def get_box(self, boxes, confs):
        confs = confs.reshape(*confs.shape[:2], -1)
        boxes = boxes.reshape(*boxes.shape[:2], -1)
        _keep = confs.argmax(-1)
        x1x2 = torch.stack([_box[:2, _select] + self.coord_x[_select] for _box, _select in zip(boxes, _keep)])
        y1y2 = torch.stack([_box[2:, _select] + self.coord_y[_select] for _box, _select in zip(boxes, _keep)])
        return torch.cat([x1x2[:,0], y1y2[:,0], x1x2[:,1], y1y2[:,1]], -1)


if __name__ == '__main__':
    heatmap = torch.ones((1, 1, 256, 256)) * (-5)
    heatmap[0, 0, 125, 10] = 20.0
    heatmap = heatmap.cuda()
    fake_data = torch.rand([2, 64, 16, 16])
    rpn_head = RPNHead()
    t = rpn_head(fake_data)
    print(t.shape)
