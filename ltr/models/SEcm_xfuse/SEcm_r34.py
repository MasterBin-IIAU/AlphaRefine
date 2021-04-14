import torch.nn as nn
import ltr.models.backbone as backbones
from ltr.models.neck.Naive_Corr import Naive_Corr
from ltr.models.neck.Depth_Corr import Depth_Corr
from ltr.models.head import bbox, corner_coarse, mask
from ltr import model_constructor


class SEcmnet(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """

    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEcmnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        assert len(head_module) == 2
        self.corner_head, self.mask_head = head_module
        self.used_layers = used_layers

        self.feat_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
        if unfreeze_layer3:
            for p in self.feature_extractor.layer3.parameters():
                p.requires_grad_(True)

    def forward(self, train_imgs, test_imgs, train_bb, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs, mode)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch """
        '''train_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        num_sequences = train_imgs.shape[-4]  # batch
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1  # 1

        # Extract backbone features
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))  # `OrderedDict{'layer3':Tensor}`
        train_feat_list = [feat for feat in train_feat_dict.values()]

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))

    def forward_test(self, test_imgs, mode='train', branches=['corner']):
        """ Forward pass of test branch. size of test_imgs is (1, batch, 3, 256, 256)"""
        # for debugging
        # from devtool import node_buffer
        # node_buffer.test_imgs = test_imgs

        output = {}
        # Extract backbone features
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1', 'layer1', 'layer2', 'layer3'])
        # Save low-level feature list
        Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([test_feat_dict['layer3']])

        # Obtain bbox prediction
        if mode == 'train':
            output['corner'] = self.corner_head(fusion_feat)
            Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list)]
            output['mask'] = self.mask_head(fusion_feat, Lfeat_list)

        elif mode == 'test':
            output['feat'] = fusion_feat
            if 'corner' in branches:
                output['corner'] = self.corner_head(fusion_feat)
            if 'mask' in branches:
                Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list)]
                output['mask'] = self.mask_head(fusion_feat, Lfeat_list)
        else:
            raise ValueError("mode should be train or test")
        return output

    def get_output(self, mode):
        if mode == 'corner':
            return self.corner_head(self.fusion_feat)
        elif mode == 'mask':
            return self.mask_head(self.fusion_feat, self.Lfeat_list)
        else:
            raise ValueError('mode should be bbox or corner or mask')

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


@model_constructor
def SEcm_resnet34_naive(backbone_pretrained=True, used_layers=('layer3',), pool_size=None, unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet34(pretrained=backbone_pretrained)

    # neck
    neck_net = Naive_Corr(pool_size=pool_size)

    # multiple heads
    corner_head = corner_coarse.Corner_Predictor(inplanes=pool_size * pool_size)  # 64
    mask_head = mask.Mask_Predictor_fine()

    # net
    net = SEcmnet(feature_extractor=backbone_net,
                  neck_module=neck_net,
                  head_module=(corner_head, mask_head),
                  used_layers=used_layers, extractor_grad=True,
                  unfreeze_layer3=unfreeze_layer3)
    return net


@model_constructor
def SEcm_resnet34_dwcorr(backbone_pretrained=True, used_layers=('layer3',), pool_size=None, unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet34(pretrained=backbone_pretrained)

    # neck
    neck_net = Depth_Corr(pool_size=pool_size)

    # multiple heads
    corner_head = corner_coarse.Corner_Predictor(inplanes=pool_size * pool_size)  # 64
    mask_head = mask.Mask_Predictor_fine()

    # net
    net = SEcmnet(feature_extractor=backbone_net,
                  neck_module=neck_net,
                  head_module=(corner_head, mask_head),
                  used_layers=used_layers, extractor_grad=True,
                  unfreeze_layer3=unfreeze_layer3)
    return net