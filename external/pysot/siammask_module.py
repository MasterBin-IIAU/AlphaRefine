# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import numpy as np

'''Attention. If we use more than 1 pysot model, we need to create new such files'''
from external.pysot.pysot.core import cfg_new
from external.pysot.pysot.models.model_builder_new import ModelBuilder
from external.pysot.pysot import build_tracker

from external.pysot.pysot import load_pretrain

torch.set_num_threads(1)


class SiamMask(object):
    def __init__(self):
        project_path_ = os.path.dirname(__file__)
        siam_model_ = 'siammask_r50_l3'
        snapshot_path_ = os.path.join(project_path_, 'experiments/%s/model.pth' % siam_model_)
        config_path_ = os.path.join(project_path_, 'experiments/%s/config.yaml' % siam_model_)
        # load config
        cfg_new.merge_from_file(config_path_)
        # create model
        model = ModelBuilder()
        # load model
        model = load_pretrain(model, snapshot_path_).cuda().eval()
        # build tracker
        self.tracker = build_tracker(model)

    def initialize(self, img, gt1):
        '''img: cv2 (BGR format), gt1: list [x1,y1,w,h] '''
        self.tracker.init(img, gt1)

    def refine(self, img, ori_bbox, VOT):
        '''img: BGR format, ori_bbox: (x1,y1,w,h) format'''
        x1, y1, w, h = ori_bbox
        '''pass ori bbox to siammask'''
        self.tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])  # (cx,cy)
        self.tracker.size = np.array([w, h])  # (w,h)
        '''use siammask to predict'''
        outputs = self.tracker.track(img, VOT)
        pred_bbox = outputs['polygon']  # list (8 points or 4 points, depends on VOT)
        if VOT:
            '''如果是要测VOT数据集，那么pred_bbox是8点标注的'''
            return np.array(pred_bbox), self.tracker.center_pos, self.tracker.size
        else:
            '''如果是测其他数据集，pred_bbox返回的是4点标注'''
            return np.array(pred_bbox)

    '''2020.4.7 为了测试VOT2020而写的'''

    def refine_mask(self, img, ori_bbox):
        '''img: BGR format, ori_bbox: (x1,y1,w,h) format'''
        x1, y1, w, h = ori_bbox
        '''pass ori bbox to siammask'''
        self.tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])  # (cx,cy)
        self.tracker.size = np.array([w, h])  # (w,h)
        '''use siammask to predict'''
        outputs = self.tracker.track(img)
        pred_mask = outputs['mask']
        target_mask = (pred_mask > 0.15)
        target_mask = target_mask.astype(np.uint8)
        return target_mask
