# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np

from external.pysot.pysot import get_axis_aligned_bbox
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip

# base tracker
from pytracking.evaluation import Tracker
# settings
from arena.GOT10k.common_path import RF_type
from arena.GOT10k.common_path import (selector_path, refine_path, sr, input_sz)

from got10k.trackers import Tracker as GOT10kTracker


class Pytracking_RF(GOT10kTracker):
    def __init__(self, rf_model_code, enable_rf=True):
        _tracker_name, _tracker_param, model_name = self._get_setting()
        model_name = model_name.format(rf_model_code)
        if not enable_rf:
            model_name = model_name.replace(RF_type.format(rf_model_code), '')
        super(Pytracking_RF, self).__init__(name=model_name)
        self.enable_rf = enable_rf
        # create tracker
        tracker_info = Tracker(_tracker_name, _tracker_param, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        self.tracker = tracker_info.tracker_class(params)

        # create Refinement module
        if self.enable_rf:
            self.RF_module = RefineModule(refine_path.format(rf_model_code), selector_path,
                                          search_factor=sr, input_sz=input_sz)

    def _get_setting(self):
        raise NotImplementedError

    def init(self, image, box):
        image = np.array(image)
        self.im_H, self.im_W, _ = image.shape
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        # Initialize base tracker
        gt_bbox_np = np.array(gt_bbox_)
        gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
        init_info = {}
        init_info['init_bbox'] = gt_bbox_torch
        _ = self.tracker.initialize(image, init_info)

        # initilize refinement module for specific video
        if self.enable_rf:
            self.RF_module.initialize(image, np.array(gt_bbox_))
        self.box = box

    def update(self, image):
        image = np.array(image)

        # track
        outputs = self.tracker.track(image)
        pred_bbox = outputs['target_bbox']

        # refine tracking results
        if self.enable_rf:
            pred_bbox = self.RF_module.refine(image, np.array(pred_bbox))

            # post processing
            x1, y1, w, h = pred_bbox.tolist()
            x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.im_H, self.im_W))  # add boundary and min size limit
            w = x2 - x1
            h = y2 - y1
            new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
            new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
            new_scale = torch.sqrt(new_target_sz.prod() / self.tracker.base_target_sz.prod())

            # update tracker's state with refined result
            self.tracker.pos = new_pos.clone()
            self.tracker.target_sz = new_target_sz
            self.tracker.target_scale = new_scale
        return pred_bbox


class DiMPsuper_RF(Pytracking_RF):
    def _get_setting(self):
        _tracker_param = 'super_dimp'
        _tracker_name = 'dimp'
        model_name = _tracker_name + '_' + _tracker_param + '{}-{}'.format(RF_type, selector_path)
        return _tracker_name, _tracker_param, model_name


class DiMP50_RF(Pytracking_RF):
    def _get_setting(self):
        _tracker_param = 'dimp50'
        _tracker_name = 'dimp'
        model_name = _tracker_name + '_' + _tracker_param + '{}-{}'.format(RF_type, selector_path)
        return _tracker_name, _tracker_param, model_name


class ECO_RF(Pytracking_RF):
    def _get_setting(self):
        _tracker_param = 'default'
        _tracker_name = 'eco'
        model_name = _tracker_name + '_' + _tracker_param + '{}-{}'.format(RF_type, selector_path)
        return _tracker_name, _tracker_param, model_name


class ATOM_RF(Pytracking_RF):
    def _get_setting(self):
        _tracker_param = 'default'
        _tracker_name = 'atom'
        model_name = _tracker_name + '_' + _tracker_param + '{}-{}'.format(RF_type, selector_path)
        return _tracker_name, _tracker_param, model_name
