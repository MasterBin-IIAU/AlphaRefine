# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import torch
import numpy as np

from external.pysot.pysot.core.config import cfg
from external.pysot.pysot.models.model_builder import ModelBuilder
from external.pysot.pysot.tracker.tracker_builder import build_tracker
from external.pysot.pysot.utils.model_load import load_pretrain
from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox
from pytracking.RF_utils import bbox_clip
from pytracking.refine_modules.refine_module import RefineModule

from arena.TrackingNet.common_path_siamrpn import *


parser = argparse.ArgumentParser(description='TrackingNet tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str,
                    help='siamrpn_r50_l234_dwxcorr, siamrpn_r50_l234_dwxcorr_otb')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')

args = parser.parse_args()
torch.set_num_threads(1)


def main():
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)
    model_name = 'siamrpn_' + RF_type; print(model_name)

    dataset_root = dataset_root_

    snapshot_path = os.path.join(project_path_, 'experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'experiments/%s/config.yaml' % args.tracker_name)

    cfg.merge_from_file(config_path)

    # create model
    model = ModelBuilder()  # a model is a Neural Network (`torch.nn.Module`)
    model = load_pretrain(model, snapshot_path).cuda().eval()

    # build tracker
    tracker = build_tracker(model)  # tracker is a object consisting of a NN and some post-processing

    # create dataset
    frames_dir = os.path.join(dataset_root,'frames')
    seq_list = sorted(os.listdir(frames_dir))

    # OPE tracking
    for v_idx, seq_name in enumerate(seq_list):
        if args.video != '':
            # test one special video
            if seq_name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        seq_frame_dir = os.path.join(frames_dir,seq_name)
        num_frames = len(os.listdir(seq_frame_dir))
        gt_file = os.path.join(dataset_root,'anno','%s.txt'%seq_name)
        gt_bbox = np.loadtxt(gt_file,dtype=np.float32,delimiter=',').squeeze()
        for idx in range(num_frames):
            frame_path = os.path.join(seq_frame_dir,'%d.jpg'%idx)
            img = cv2.imread(frame_path)
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                H,W,_ = img.shape
                '''Initialize'''
                tracker.init(img, gt_bbox_)
                '''##### initilize refinement module for specific video'''
                RF_module.initialize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.array(gt_bbox_))
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)

            else:
                '''Track'''
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                '''##### refine tracking results #####'''
                pred_bbox = RF_module.refine(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.array(pred_bbox))
                x1, y1, w, h = pred_bbox.tolist()
                '''add boundary and min size limit'''
                x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                w = x2 - x1
                h = y2 - y1
                pred_bbox = np.array([x1,y1,w,h])
                tracker.center_pos = np.array([x1 + w / 2, y1 + h / 2])
                tracker.size = np.array([w, h])
                pred_bboxes.append(pred_bbox)

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(seq_name, img)
                cv2.waitKey(1)

        toc /= cv2.getTickFrequency()
        # save results
        model_path = os.path.join(save_dir, 'trackingnet', model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(seq_name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, seq_name, toc, idx / toc))


if __name__ == '__main__':
    main()
