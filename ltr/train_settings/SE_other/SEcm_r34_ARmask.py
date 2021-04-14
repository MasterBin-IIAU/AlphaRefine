"""training a scale estimator with 2 branches"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data.distributed import DistributedSampler

from ltr.dataset.ardataset import Got10k, Lasot, MSCOCOSeq, ImagenetVID, ImagenetDET, Youtube_VOS, Saliency
from ltr.data.ardata import SEprocessing, SEsampler, LTRLoader
import ltr.data.ardata.transforms as dltransforms
from ltr import actors
from ltr.trainers import LTRTrainer
from ltr.models.loss.iou_loss import IOULoss_mean
import ltr.models.SEx_beta.SEcm_r34 as SEx


def run(settings):

    # Most common settings are assigned in the settings struct
    settings.description = 'Settings of SEcm module'
    '''!!! some important hyperparameters !!!'''
    settings.batch_size = 32  # Batch size
    settings.search_area_factor = 2.0  # Image patch size relative to target size
    settings.feature_sz = 16  # Size of feature map
    settings.output_sz = settings.feature_sz * 16  # Size of input image patches
    settings.used_layers = ['layer3']
    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 0.25}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}
    settings.max_gap = 50
    settings.sample_per_epoch = 4000  # 由于batchsize比原来更小了,所以我把这个值又翻了一倍

    '''others'''
    settings.print_interval = 100  # How often to print loss and other info
    settings.num_workers = 4  # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)

    '''##### Prepare data for training and validation #####'''
    '''1. build trainning dataset and dataloader'''
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
    # -bbox and corner datasets
    # got_10k_train = Got10k(settings.env.got10k_dir, split='train')
    # lasot_train = Lasot(split='train')
    coco_train = MSCOCOSeq()
    imagenet_vid = ImagenetVID()
    imagenet_det = ImagenetDET()

    # -mask datasets
    youtube_vos = Youtube_VOS()
    # saliency = Saliency()

    # The sampler for training
    '''Build training dataset. focus on "__getitem__" and "__len__"'''
    dataset_train = SEsampler.SEMaskSampler([coco_train,imagenet_vid,imagenet_det,youtube_vos],
                                        [1,1,1,2],
                                        samples_per_epoch= settings.sample_per_epoch * settings.batch_size,
                                        max_gap=settings.max_gap,
                                        processing=data_processing_train)

    # The loader for training
    '''using distributed sampler'''
    train_sampler = DistributedSampler(dataset_train)
    '''"sampler" is exclusive with "shuffle"'''
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             drop_last=True, stack_dim=1, sampler=train_sampler, pin_memory=False)


    '''2. build validation dataset and dataloader'''
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

    #The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network
    net = SEx.SEcm_resnet34(backbone_pretrained=True,
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
    actor = actors.SEcm_Actor(net=net, objective=objective)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # load specified pre-trained parameter
    load_pretrained = False
    if hasattr(settings, 'pretrained') and settings.pretrained is not None:
        trainer.load_pretrained(settings.pretrained)
        load_pretrained = True

    # launch training process
    trainer.train(40, load_latest=not load_pretrained, fail_safe=False)
