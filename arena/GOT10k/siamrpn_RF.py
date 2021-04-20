# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
from external.pysot.pysot.core import cfg
from external.pysot.pysot.models import ModelBuilder
from external.pysot.pysot.tracker.tracker_builder import build_tracker
from external.pysot.pysot import get_axis_aligned_bbox
from external.pysot.pysot import load_pretrain

from pytracking.RF_utils import bbox_clip
from pytracking.refine_modules.refine_module import RefineModule

from got10k.trackers import Tracker as GOT10kTracker
from arena.GOT10k.common_path_siamrpn import *


class SiamRPNpp_RF(GOT10kTracker):
    def __init__(self, rf_model_code, enable_rf=True):
        model_name = 'siamrpn_' + RF_type.format(rf_model_code)
        if not enable_rf:
            model_name = model_name.replace(RF_type.format(rf_model_code), '')
        super(SiamRPNpp_RF, self).__init__(name=model_name)
        self.enable_rf = enable_rf
        # create tracker
        snapshot_path = os.path.join(project_path_, 'experiments/%s/model.pth' % siam_model_)
        config_path = os.path.join(project_path_, 'experiments/%s/config.yaml' % siam_model_)
        cfg.merge_from_file(config_path)
        model = ModelBuilder()  # a sub-class of `torch.nn.Module`
        model = load_pretrain(model, snapshot_path).cuda().eval()
        self.tracker = build_tracker(model)  # tracker is a object consisting of NN and some post-processing

        # create refinement module
        if self.enable_rf:
            self.RF_module = RefineModule(refine_path.format(rf_model_code), selector_path,
                                          search_factor=sr, input_sz=input_sz)

    def init(self, image, box):
        image = np.array(image)
        self.im_H, self.im_W, _ = image.shape
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        self.tracker.init(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), gt_bbox_)  # image is RGB, does siamrpn take BGR image?

        if self.enable_rf:
            self.RF_module.initialize(image, np.array(gt_bbox_))
        self.box = box

    def update(self, image):
        image = np.array(image)
        outputs = self.tracker.track(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # image is RGB, does siamrpn take BGR image?
        pred_bbox = outputs['bbox']

        # refine tracking results
        if self.enable_rf:
            pred_bbox = self.RF_module.refine(image, np.array(pred_bbox))
            x1, y1, w, h = pred_bbox.tolist()

            x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.im_H, self.im_W))  # add boundary and min size limit
            w = x2 - x1
            h = y2 - y1
            pred_bbox = np.array([x1, y1, w, h])
            self.tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])
            self.tracker.size = np.array([w, h])
        return pred_bbox
