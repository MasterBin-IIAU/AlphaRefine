import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = []

        stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5),  # s2
            nn.ReLU(inplace=True)
        )
        self.features.append(stage1)

        stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )# s4
        self.features.append(stage2)

        stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )  # s8)
        self.features.append(stage3)

        stage4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)  # s16
        )
        self.features.append(stage4)
        self.features = nn.ModuleList(self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, layers=None):
        out_feat = []

        x = self.features[0](x)
        out_feat.append(x)

        x = self.features[1](x)
        out_feat.append(x)

        x = self.features[2](x)
        out_feat.append(x)

        x = self.features[3](x)
        out_feat.append(x)

        return out_feat


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        import copy
        official_keys = list(state_dict.keys())
        state_dict_ = copy.deepcopy(model.state_dict())
        for idx, k in enumerate(state_dict_):
            assert state_dict_[k].shape == state_dict[official_keys[idx]].shape
            state_dict_[k] = state_dict[official_keys[idx]]
        model.load_state_dict(state_dict_, strict=True)
    return model
