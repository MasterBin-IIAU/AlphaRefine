# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import sys
sys.path.insert(0, '/home/zxy/Desktop/AlphaRefine/pysot')

import cv2
import torch
import numpy as np
from external.pysot import cfg
from external.pysot import ModelBuilder
from external.pysot import build_tracker
from external.pysot import get_axis_aligned_bbox
from external.pysot import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

'''Refine module'''
from pytracking.refine_module import RefineModule
from pytracking.evaluation import Tracker
from siammask_module import siammask
'''RF utils'''
from pytracking.RF_utils import bbox_clip
'''common paths'''
from common_path_siamrpn import *

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str,
                    help='siamrpn_r50_l234_dwxcorr, siamrpn_r50_l234_dwxcorr_otb')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
# parser.add_argument('--refine_method',default='RF',type=str,help='RF, iou_net, mask')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--gpu_id', default=1, type=int)

args = parser.parse_args()

torch.set_num_threads(1)
torch.cuda.set_device(args.gpu_id) # set GPU id


# refine_path = "/home/zxy/Desktop/AlphaRefine/AlphaRefine_CVPR21/pytracking/networks/SEbcmnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm-8gpu/SEbcmnet_ep0040.pth.tar"  # siamrpn_RF_BCM8GPU_corner  # RF_CrsM_ARv1

# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEc/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEc/SEcnet_ep0035.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEc_woNL/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_woNL/SEcmnet_ep0040_again.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_woNL/SEcmnet_ep0040-again2.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEc_again/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEc_again/SEcnet_ep0035.pth.tar"

# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm/SEbcmnet_ep0040.pth.tar"

# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEc1x1m/SEcmnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEc1x1m_woNL/SEcmnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs1x1_woNL/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs_woNL/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs_light/SEcnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs_rmNL/SEcnet_ep0040.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm2/SEb2cm/SEbcmnet_ep0040-a.pth.tar"  # siamrpn_RF_B2CM_a_1
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm2/SEb2cm/SEbcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm2/SEb2cm/SEbcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm2/SEb2cm/SEbcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm2/SEb2cm/SEbcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm/SEbcmnet_ep0040-a.pth.tar"  # siamrpn_RF_BM_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm/SEbcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm/SEbcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm/SEbcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm/SEbcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEb/SEbcmnet_ep0040-b.pth.tar"  # siamrpn_RF_B_b_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEb/SEbcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEb/SEbcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEb/SEbcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm_v2/SEbcmnet_ep0040-a.pth.tar"  # siamrpn_RF_B2CM_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm_v2/SEbcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm_v2/SEbcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm_v2/SEbcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm2/SEbm_v2/SEbcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040.pth.tar"  # siamrpn_RF_CM_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040.pth-d.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040-e.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040-again.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040-again2.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_cf/SEcmNet_ep0040-a.pth.tar"  # siamrpn_RF_CfM_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_cf/SEcmNet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_cf/SEcmNet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_cf/SEcmNet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_cf/SEcmNet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm/SEcmnet_ep0040.pth.tar"  # siamrpn_RF_CrsM_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm/SEcmnet_ep0040-e.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcmnet_ep0040-c/SEcrsm/SEcmnet_ep0040-d.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm_woNL/SEcmnet_ep0040.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_woNL/SEcmnet_ep0040.pth.tar"  # siamrpn_RF_CrsM_woNL_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_woNL/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_woNL/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_woNL/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_woNL/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_rmNL_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_iouloss/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_rmNL_iouloss_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_iouloss/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_iouloss/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_iouloss/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_iouloss/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrs2m_rmNL/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_Crs2M_rmNL_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrs2m_rmNL/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrs2m_rmNL/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrs2m_rmNL/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrs2m_rmNL/SEcmnet_ep0040-e.pth.tar"


## for pixel-corr response visualization
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm/SEcmnet_ep0040-b-0.634.pth.tar"  # siamrpn_RF_CrsMAgain_0


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_corr/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_NaiveCorr_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_corr/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_corr/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_corr/SEcmnet_ep0040-e.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_corr/SEcmnet_ep0040-d.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs/SEcnet_ep0040.pth.tar"  # siamrpn_RF_CrsM_rmNL_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs/SEcnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs/SEcnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs/SEcnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEc/SEcrs/SEcnet_ep0040-e.pth.tar"


# dcorr on SEcrs
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_dcorr/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_rmNL_dcorr_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_dcorr/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_dcorr/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_dcorr/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL_dcorr/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm3/SEbcm3_m2bloss/SEbcmnet_ep0040-a.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm3/SEbcm3_m2bloss/SEbcmnet_ep0040-g-extreme-only.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm3/SEbcm3_m2bloss/SEbcmnet_ep0040-g-extreme-only.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-a.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm3/SEbcm3_m2bloss/SEbcmnet_ep0040-d-extrame.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_R34_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-a.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-b.pth.tar"  # RF_CrsM_R34woPr_b
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-d.pth.tar"
refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr/SEcmnet_ep0040-a.pth.tar"  # RF_CrsM_R34SR15_a
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr_fcn/SEcmnet_ep0040-a.pth.tar"  # RF_CrsM_R34SR15FCN_a
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr_fcn/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr_fcn/SEcmnet_ep0040-c.pth.tar"

# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_25sr/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_R34SR25_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_25sr/SEcmnet_ep0040-b.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r18/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_R18_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r18/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r18/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r18/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r18/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_mbv2/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_MBV2_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_mbv2/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_mbv2/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_mbv2/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_mbv2/SEcmnet_ep0040-e.pth.tar"


# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_efb0/SEcmnet_ep0040-a.pth.tar"  # siamrpn_RF_CrsM_EFB0_a_0
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_efb0/SEcmnet_ep0040-b.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_efb0/SEcmnet_ep0040-c.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_efb0/SEcmnet_ep0040-d.pth.tar"
# refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_efb0/SEcmnet_ep0040-e.pth.tar"

selector_path = 0; branches = ['corner', 'mask'][0:1]
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
# selector_path = '/home/zxy/Desktop/AlphaRefine/AlphaRefine_CVPR21/pytracking/networks/Branch_Selector_ep0030.pth.tar'

refine_method = 'RF_CrsM_R34woPr_e'
# refine_method = 'RF_CrsM_ARv1'
# refine_method = 'RF_speed_test'


def main():
    # refine_method = args.refine_method
    model_name = 'siamrpn_' + refine_method
    model_path = '/'
    snapshot_path = os.path.join(project_path_, 'experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'experiments/%s/config.yaml' % args.tracker_name)

    cfg.merge_from_file(config_path)
    dataset_root = dataset_root_

    # create model
    '''a model is a Neural Network.(a torch.nn.Module)'''
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, snapshot_path).cuda().eval()

    # build tracker
    '''a tracker is a object, which consists of not only a NN but also some post-processing'''
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    '''##### build a refinement module #####'''
    if 'RF' in refine_method:
        RF_module = RefineModule(refine_path, selector_path, branches=branches, search_factor=sr, input_sz=input_sz)

    elif refine_method == 'iou_net':
        RF_info = Tracker('iou_net', 'iou_net_dimp', None)
        RF_params = RF_info.get_parameters()
        RF_params.visualization = False
        RF_params.debug = False
        RF_params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        RF_module = RF_info.tracker_class(RF_params)

    elif refine_method == 'mask':
        RF_module = siammask()
    else:
        raise ValueError ("refine_method should be 'RF' or 'iou' or 'mask' ")
    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    H,W,_ = img.shape
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    '''##### initilize refinement module for specific video'''
                    if 'RF' in refine_method:
                        RF_module.initialize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                         np.array(gt_bbox_))
                    elif refine_method == 'iou_net':
                        gt_bbox_np = np.array(gt_bbox_)
                        gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
                        init_info = {}
                        init_info['init_bbox'] = gt_bbox_torch
                        RF_module.initialize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),init_info)
                    elif refine_method == 'mask':
                        RF_module.initialize(img,np.array(gt_bbox_))
                    else:
                        raise ValueError ("refine_method should be 'RF' or 'RF_mask' or 'iou_net' or 'mask' ")
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    '''##### refine tracking results #####'''
                    if 'RF' in refine_method or refine_method == 'iou_net':
                        pred_bbox = RF_module.refine(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                                        np.array(pred_bbox))
                        x1, y1, w, h = pred_bbox.tolist()
                        '''add boundary and min size limit'''
                        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                        w = x2 - x1
                        h = y2 - y1
                        pred_bbox = np.array([x1, y1, w, h])
                        '''pass new state back to base tracker'''
                        tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])
                        tracker.size = np.array([w, h])
                    elif refine_method == 'mask':
                        pred_bbox, center_pos, size = RF_module.refine(img,np.array(pred_bbox),VOT=True)
                        # boundary and min size limit have been included in "refine"
                        '''pass new state back to base tracker'''
                        '''pred_bbox is a list with 8 elements'''
                        tracker.center_pos = center_pos
                        tracker.size = size
                    else:
                        raise ValueError ('refine_method should be RF or iou or mask')
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if refine_method == 'mask':
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(save_dir, args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if video.name+'.txt' in os.listdir(model_path):
                continue
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    H,W,_ = img.shape
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    '''##### initilize refinement module for specific video'''
                    if 'RF' in refine_method:
                        RF_module.initialize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                             np.array(gt_bbox_))
                    elif refine_method == 'iou_net':
                        gt_bbox_np = np.array(gt_bbox_)
                        gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
                        init_info = {}
                        init_info['init_bbox'] = gt_bbox_torch
                        RF_module.initialize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), init_info)
                    elif refine_method == 'mask':
                        RF_module.initialize(img, np.array(gt_bbox_))
                    else:
                        raise ValueError ("refine_method should be 'RF' or 'iou' or 'mask' ")
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    '''##### refine tracking results #####'''
                    if 'RF' in refine_method or refine_method == 'iou_net':
                        pred_bbox = RF_module.refine(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                     np.array(pred_bbox))
                    elif refine_method == 'mask':
                        pred_bbox = RF_module.refine(img, np.array(pred_bbox),VOT=False)
                    else:
                        raise ValueError ("refine_method should be 'RF' or 'iou' or 'mask' ")
                    x1, y1, w, h = pred_bbox.tolist()
                    '''add boundary and min size limit'''
                    x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                    w = x2 - x1
                    h = y2 - y1
                    pred_bbox = np.array([x1,y1,w,h])
                    tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])
                    tracker.size = np.array([w, h])

                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(save_dir, args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(save_dir, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(save_dir, args.dataset, model_name+'_'+str(selector_path))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
