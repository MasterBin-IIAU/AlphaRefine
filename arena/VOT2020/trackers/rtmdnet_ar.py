import numpy as np

from external.RT_MDNet.RT_MDNet import RT_MDNet
from pytracking.refine_modules.refine_module_vot20 import RefineModule

from arena.VOT2020.utils import rect_from_mask, bbox_clip
from arena.VOT2020.common_path import refine_path, sr, input_sz


class TrackerAR(object):
    def __init__(self, threshold=0.65):
        self.THRES = threshold
        '''create tracker'''
        self.base_tracker = tracker = RT_MDNet()  # a tracker is a object consisting of not only a NN and some post-processing

        '''Alpha-Refine'''
        self.alpha = RefineModule(refine_path, sr, input_sz=input_sz)

    def initialize(self, img_RGB, mask):
        region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        _ = self.base_tracker.initialize_seq(img_RGB, gt_bbox_np)

        '''initilize refinement module for specific video'''
        self.alpha.initialize(img_RGB, np.array(gt_bbox_np))

    def track(self, img_RGB):
        """ tracking pipeline """

        '''Step0: run base tracker'''
        pred_bbox = self.base_tracker.track(img_RGB)

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
        self.base_tracker = tracker = RT_MDNet()  # a tracker is a object consisting of not only a NN and some post-processing

    def initialize(self, img_RGB, mask):
        region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        _ = self.base_tracker.initialize_seq(img_RGB, gt_bbox_np)

    def track(self, img_RGB):
        """ tracking pipeline """

        '''Step0: run base tracker'''
        pred_bbox = self.base_tracker.track(img_RGB)

        '''Step1: Post-Process'''
        x1, y1, w, h = pred_bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1

        bbox_new = [x1,y1,w,h]

        '''Step2: Mask report'''
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
