# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox

from external.RT_MDNet.RT_MDNet import RT_MDNet
from pytracking.refine_modules.refine_module import RefineModule

from got10k.trackers import Tracker as GOT10kTracker
from external.RT_MDNet.MDNet_utils import bbox_clip
from arena.GOT10k.common_path import *


class RTMDNet_RF(GOT10kTracker):
    def __init__(self, rf_model_code, enable_rf=True):
        model_name = 'RTMDNet' + '{}-{}'.format(RF_type.format(rf_model_code), selector_path)
        if not enable_rf:
            model_name = model_name.replace(RF_type.format(rf_model_code), '')
        super(RTMDNet_RF, self).__init__(name=model_name)
        self.enable_rf = enable_rf

        self.tracker = RT_MDNet()
        if self.enable_rf:
            self.RF_module = RefineModule(refine_path.format(rf_model_code), selector_path,
                                          search_factor=sr, input_sz=input_sz)

    def init(self, image, box):
        image = np.array(image)
        self.im_H, self.im_W, _ = image.shape
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        # initialize tracker
        self.tracker.initialize_seq(image, np.array(gt_bbox_))
        # initilize refine module
        if self.enable_rf:
            self.RF_module.initialize(image, np.array(gt_bbox_))
        self.box = box

    def update(self, image):
        image = np.array(image)
        pred_bbox = self.tracker.track(image)

        if self.enable_rf:
            # refine tracking results
            pred_bbox = self.RF_module.refine(image, np.array(pred_bbox))
            pred_bbox = bbox_clip(pred_bbox, (self.im_H, self.im_W))  # boundary and size limit
            '''update state'''
            self.tracker.target_bbox = pred_bbox.copy()
        return pred_bbox
