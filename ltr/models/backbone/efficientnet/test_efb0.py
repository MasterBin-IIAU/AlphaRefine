import torch
import tqdm
from ltr.models.backbone.efficientnet.model4ar import efficientnet
from ltr.models.backbone.resnet import resnet50, resnet34

# torch.backends.cudnn.benchmark = True

net = efficientnet('efficientnet-b0').cuda()
# net = resnet50(['conv1', 'layer1', 'layer2', 'layer3']).cuda()
# net = resnet34(['conv1', 'layer1', 'layer2', 'layer3']).cuda()
batch_size = 1
with torch.no_grad():
    for i in tqdm.tqdm(range(5000)):
        fake_data = torch.rand(batch_size, 3, 224, 224).cuda()
        out = net(fake_data)
        if isinstance(out, dict):
            out = [o for k, o in out.items()]
        # print(out[-1].shape)
