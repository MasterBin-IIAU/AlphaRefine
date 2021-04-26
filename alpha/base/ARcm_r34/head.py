import torch.nn as nn
import torch
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class Mask_Predictor_fine(nn.Module):
    """ Similar to SiamMask, merge the low level feature """
    def __init__(self):
        super(Mask_Predictor_fine, self).__init__()
        self.v0 = nn.Sequential(
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v1 = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v2 = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h2 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h0 = nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, corr_feat, Lfeat):
        """
        corr_feat是经过correlation和Non-local处理后的特征, Lfeat是backbone提取出的底层特征
        Args:
            corr_feat: (batch,64,16,16)
            Lfeat: (batch,64,128,128), (batch,256,64,64), (batch,512,32,32)
        """

        out = self.post0(F.interpolate(self.h2(corr_feat),size=(32,32),mode='bilinear')+self.v2(Lfeat[2]))  # (b,32,32,32)+(b,32,32,32) --> (b,16,32,32)
        out = self.post1(F.interpolate(self.h1(out),size=(64,64),mode='bilinear')+self.v1(Lfeat[1]))  # (b,16,64,64)+(b,16,64,64) --> (b,4,64,64)
        out = self.post2((F.interpolate(self.h0(out), size=(128, 128),mode='bilinear') + self.v0(Lfeat[0])).contiguous())
        out = torch.sigmoid(F.interpolate(out,size=(256,256),mode='bilinear'))  # (b,1,256,256)
        return out


class CornerPredictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256):
        super(CornerPredictor, self).__init__()

        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel)
        self.conv2_tl = conv(channel, channel // 2)
        self.conv3_tl = conv(channel // 2, channel // 4)
        self.conv4_tl = conv(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel)
        self.conv2_br = conv(channel, channel // 2)
        self.conv3_br = conv(channel // 2, channel // 4)
        self.conv4_br = conv(channel // 4, channel // 8)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

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
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):

        #top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

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

    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.output_sz * self.output_sz)), dim=1)  # (batch, output_sz * output_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


if __name__ == '__main__':
    heatmap = torch.ones((1, 1, 256, 256)) * (-5)
    heatmap[0, 0, 125, 10] = 20.0
    heatmap = heatmap.cuda()
    corner_predictor = CornerPredictor()
