# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import cv2
import torch
import numpy as np

from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip

torch.set_num_threads(1)


class DBLoader(object):
    """ Debug Data Loader """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gt_file = os.path.join(self.data_dir, 'groundtruth.txt')
        self.curr_idx = 0
        self.im_paths = glob.glob(os.path.join(self.data_dir, 'color/*.jpg'))
        self.im_paths.sort()
        self.init_box = self.get_init_box()

    def get_init_box(self):
        im_path = self.im_paths[0]
        first_frame = cv2.imread(im_path)
        init_box = cv2.selectROI(os.path.basename(self.data_dir), first_frame, False, False)
        return np.array(init_box)

    def region(self):
        return self.init_box

    def frame(self):
        im_path = self.im_paths[self.curr_idx] if self.curr_idx < len(self.im_paths) else None
        print('pumping {}'.format(im_path))
        self.curr_idx += 1
        return im_path, None


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def get_dimp(img, init_box, model_path):
    """ set up DiMPsuper as the base tracker """
    from pytracking.parameter.dimp.super_dimp_demo import parameters
    from pytracking.tracker.dimp.dimp import DiMP

    params = parameters(model_path)
    params.visualization = True
    params.debug = False
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = DiMP(params)

    H, W, _ = img.shape
    cx, cy, w, h = get_axis_aligned_bbox(np.array(init_box))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    '''Initialize'''
    gt_bbox_np = np.array(gt_bbox_)
    gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
    init_info = {}
    init_info['init_bbox'] = gt_bbox_torch
    tracker.initialize(img, init_info)

    return tracker


def get_ar(img, init_box, ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))
    return RF_module


def demo(base_path, ar_path, data_dir):
    debug_loader = DBLoader(data_dir=data_dir)

    handle = debug_loader
    init_box = handle.region()
    imagefile, _ = handle.frame()
    img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    H, W, _ = img.shape

    """ Step 1: set up base tracker and Alpha-Refine """
    tracker = get_dimp(img, init_box, base_path)
    RF_module = get_ar(img, init_box, ar_path)

    # OPE tracking
    while True:
        imagefile, _ = handle.frame()
        if not imagefile:
            break
        img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        """ Step 2: base tracker prediction """
        # track with base tracker
        outputs = tracker.track(img)
        pred_bbox = outputs['target_bbox']

        """ Step 3: refine tracking results with Alpha-Refine """
        pred_bbox = RF_module.refine(img, np.array(pred_bbox))

        """ Step 4: update base tracker's state with refined result """
        x1, y1, w, h = pred_bbox.tolist()
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())

        tracker.pos = new_pos.clone()
        tracker.target_sz = new_target_sz
        tracker.target_scale = new_scale

        # visualization
        pred_bbox = list(map(int, pred_bbox))
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(_img, (pred_bbox[0], pred_bbox[1]),
                      (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
        cv2.imshow('', _img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit(0)


if __name__ == '__main__':
    # path to video sequence - any directory with series of images will be OK
    data_dir = 'data/RGBD19/box_room_occ_1'

    # path to model_file of base tracker - model can be download from:
    # https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv
    base_path = '/home/zxy/Desktop/AR_Maintaince/AlphaRefine/pytracking/networks/super_dimp.pth.tar'

    # path to model_file of Alpha-Refine - the model can be download from
    # https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view
    ar_path = '/home/zxy/Desktop/AR_Maintaince/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet_ep0040-c.pth.tar'

    demo(base_path, ar_path, data_dir)
