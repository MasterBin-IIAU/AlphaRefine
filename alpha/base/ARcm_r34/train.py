"""training a scale estimator with 2 branches"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data.distributed import DistributedSampler

from ltr.dataset.ardataset import Got10k, Lasot, MSCOCOSeq, ImagenetVID, ImagenetDET, Youtube_VOS, Saliency
from ltr.data.ardata import SEprocessing, SEsampler, LTRLoader
import ltr.data.ardata.transforms as dltransforms
from ltr.trainers import ARTrainer

from .actor import SEcm_Actor
from .model import SEcm_resnet34 as SEx
from .settings import param_setup


def run(settings):
    param_setup(settings)

    ##### Prepare data for training and validation #####

    # ====  1. build trainning dataset and dataloader ====

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)
    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                       std=settings.normalize_std)])
    # Data processing to do on the training pairs
    '''Data_process class. In SEMaskProcessing, we use zero-padding for images and masks.'''
    data_processing_train = SEprocessing.SEMaskProcessing(search_area_factor=settings.search_area_factor,
                                                          output_sz=settings.output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint)

    # Train datasets
    # - bbox and corner datasets
    got_10k_train = Got10k(settings.env.got10k_dir, split='train')
    lasot_train = Lasot(split='train')
    coco_train = MSCOCOSeq()
    imagenet_vid = ImagenetVID()
    imagenet_det = ImagenetDET()

    # - mask datasets
    youtube_vos = Youtube_VOS()
    saliency = Saliency()

    # The sampler for training
    # Build training dataset. focus on "__getitem__" and "__len__"
    dataset_train = SEsampler.SEMaskSampler([lasot_train,got_10k_train,coco_train,imagenet_vid,imagenet_det,youtube_vos,saliency],
                                            [1, 1, 1, 1, 1, 2, 3],
                                            samples_per_epoch= settings.sample_per_epoch * settings.batch_size,
                                            max_gap=settings.max_gap,
                                            processing=data_processing_train)

    # The loader for training
    # using distributed sampler
    train_sampler = DistributedSampler(dataset_train)
    # "sampler" is exclusive with "shuffle"
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             drop_last=True, stack_dim=1, sampler=train_sampler, pin_memory=False)

    # ==== 2. build validation dataset and dataloader ====

    lasot_test = Lasot(split='test')
    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])
    # Data processing to do on the validation pairs
    data_processing_val = SEprocessing.SEMaskProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_val,
                                                        joint_transform=transform_joint)
    # The sampler for validation
    dataset_val = SEsampler.SEMaskSampler([lasot_test], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                          processing=data_processing_val)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network
    net = SEx(backbone_pretrained=True,
              used_layers=settings.used_layers,
              pool_size=int(settings.feature_sz / 2),
              unfreeze_layer3=True)

    # wrap network to distributed one
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[settings.local_rank], find_unused_parameters=True)

    # Set objective
    objective = {}
    objective['corner'] = nn.MSELoss()  # take average of all elements
    objective['mask'] = nn.BCELoss()  # Basic BCE Loss

    # Create actor, which wraps network and objective
    actor = SEcm_Actor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=settings.learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Create trainer
    trainer = ARTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # load specified pre-trained parameter
    pretrained_loaded = False
    if hasattr(settings, 'pretrained') and settings.pretrained is not None:
        trainer.load_pretrained(settings.pretrained)
        pretrained_loaded = True
    load_latest = not pretrained_loaded  # if the specified ckpt has been loaded, the latest ckpt will not be resumed.

    # launch training process
    trainer.train(40, load_latest=load_latest, fail_safe=False, save_interval=settings.save_interval)
