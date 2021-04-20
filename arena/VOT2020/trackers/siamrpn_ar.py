import os

import numpy as np

from pytracking.refine_modules.refine_module_vot20 import RefineModule

from arena.VOT2020.utils import rect_from_mask, bbox_clip
from arena.VOT2020.common_path import refine_path, sr, input_sz

from external.pysot.pysot.models import ModelBuilder
from external.pysot.pysot.tracker.tracker_builder import build_tracker
from external.pysot.pysot import load_pretrain

from external.pysot.pysot.core import cfg
_project_path = os.path.join(os.path.dirname(__file__), '../../../pysot')
_siam_model = 'siamrpn_r50_l234_dwxcorr'
snapshot_path = os.path.join(_project_path, 'experiments/%s/model.pth' % _siam_model)
config_path = os.path.join(_project_path, 'experiments/%s/config.yaml' % _siam_model)
cfg.merge_from_file(config_path)


class TrackerAR(object):
    def __init__(self, threshold=0.65):
        self.THRES = threshold
        '''create tracker'''
        # create model
        model = ModelBuilder()  # a model is a Neural Network.(a torch.nn.Module)
        model = load_pretrain(model, snapshot_path).cuda().eval()
        self.base_tracker = build_tracker(model)  # a tracker is a object consisting of not only a NN and some post-processing

        '''Alpha-Refine'''
        self.alpha = RefineModule(refine_path, sr, input_sz=input_sz)

    def initialize(self, img_RGB, mask):
        region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        _ = self.base_tracker.init(img_RGB, region)

        '''initilize refinement module for specific video'''
        self.alpha.initialize(img_RGB, np.array(gt_bbox_np))

    def track(self, img_RGB):
        """ tracking pipeline """

        '''Step0: run base tracker'''
        outputs = self.base_tracker.track(img_RGB)
        pred_bbox = outputs['bbox']

        '''Step1: Post-Process'''
        x1, y1, w, h = pred_bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1

        bbox_new = [x1,y1,w,h]

        '''Step2: Mask report'''
        pred_mask = self.alpha.refine(img_RGB, np.array(bbox_new))
        final_mask = (pred_mask > self.THRES).astype(np.uint8)

        return bbox_new, final_mask


class BaseTracker(object):
    def __init__(self, threshold=0.65):
        self.THRES = threshold
        '''create tracker'''
        # create base tracker
        model = ModelBuilder()  # a model is a Neural Network.(a torch.nn.Module)
        model = load_pretrain(model, snapshot_path).cuda().eval()
        self.base_tracker = build_tracker(model)  # a tracker is a object consisting of not only a NN and some post-processing

    def initialize(self, img_RGB, mask):
        region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        _ = self.base_tracker.init(img_RGB, region)

    def track(self, img_RGB):
        """ tracking pipeline """

        '''Step0: run base tracker'''
        outputs = self.base_tracker.track(img_RGB)
        pred_bbox = outputs['bbox']

        '''Step1: Post-Process'''
        x1, y1, w, h = pred_bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1

        bbox_new = [x1,y1,w,h]

        final_mask = self._mask_from_rect(bbox_new, [self.W, self.H])

        return bbox_new, final_mask

    def _mask_from_rect(self, rect, output_sz):
        '''
        create a binary mask from a given rectangle
        rect: axis-aligned rectangle [x0, y0, width, height]
        output_sz: size of the output [width, height]
        '''
        mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
        x0 = max(int(round(rect[0])), 0)
        y0 = max(int(round(rect[1])), 0)
        x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
        y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
        mask[y0:y1, x0:x1] = 1
        return mask
