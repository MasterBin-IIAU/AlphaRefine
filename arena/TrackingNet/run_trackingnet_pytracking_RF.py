# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import cv2
import torch
import numpy as np
from pysot.pysot.utils.bbox import get_axis_aligned_bbox

from arena.TrackingNet.common_path import save_dir, dataset_root_

###################################################
'''atom'''
# tracker_param_ = 'default'
# tracker_name_ = 'atom'
###################################################
'''dimp'''
# tracker_param_ = 'super_dimp'
tracker_param_ = 'dimp50'
tracker_name_ = 'dimp'
###################################################
'''eco'''
# tracker_param_ = 'default'
# tracker_name_ = 'eco'
###################################################

from pytracking.refine_modules.refine_module import RefineModule
from arena.TrackingNet.common_path import refine_path, selector_path, sr, input_sz, RF_type

'''Refine module & Pytracking trackers'''
from pytracking.evaluation import Tracker
from pytracking.RF_utils import bbox_clip

parser = argparse.ArgumentParser(description='TrackingNet tracking')
parser.add_argument('--vis', action='store_true', default=False,
                    help='whether to visualzie result')
parser.add_argument('--debug', action='store_true', default=False,
                    help='whether to debug'),
parser.add_argument('--video', default='', type=str,
                    help='eval one special video'),
parser.add_argument('--tracker_name', default=tracker_name_, type=str,
                    help='name of tracker for pytracking tracker'),
parser.add_argument('--tracker_param', default=tracker_param_, type=str,
                    help='name of param for pytracking tracker')

args = parser.parse_args()
torch.set_num_threads(1)


def main():
    # load config
    model_name = args.tracker_name + '_' + args.tracker_param + '_' + RF_type; print(model_name)
    dataset_root = dataset_root_

    # create tracker
    tracker_info = Tracker(args.tracker_name, args.tracker_param, None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = args.debug
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    # setup refine module
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)

    # create dataset
    frames_dir = os.path.join(dataset_root, 'frames')
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
        seq_frame_dir = os.path.join(frames_dir, seq_name)
        num_frames = len(os.listdir(seq_frame_dir))
        gt_file = os.path.join(dataset_root, 'anno', '%s.txt' % seq_name)
        gt_bbox = np.loadtxt(gt_file, dtype=np.float32, delimiter=',').squeeze()
        for idx in range(num_frames):
            frame_path = os.path.join(seq_frame_dir, '%d.jpg' % idx)
            img = cv2.imread(frame_path)
            '''get RGB format image'''
            img_RGB = img[:, :, ::-1].copy()  # BGR --> RGB
            tic = cv2.getTickCount()
            if idx == 0:
                H, W, _ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                '''Initialize'''
                gt_bbox_np = np.array(gt_bbox_)
                gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
                init_info = {}
                init_info['init_bbox'] = gt_bbox_torch
                _ = tracker.initialize(img_RGB, init_info)
                '''##### initilize refinement module for specific video'''
                RF_module.initialize(img_RGB, np.array(gt_bbox_))
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)

            else:
                '''Track'''
                outputs = tracker.track(img_RGB)
                pred_bbox = outputs['target_bbox']
                '''##### refine tracking results #####'''
                pred_bbox = RF_module.refine(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                             np.array(pred_bbox))
                x1, y1, w, h = pred_bbox.tolist()
                '''add boundary and min size limit'''
                x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                w = x2 - x1
                h = y2 - y1
                new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
                new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
                new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())
                ##### update
                tracker.pos = new_pos.clone()
                tracker.target_sz = new_target_sz
                tracker.target_scale = new_scale

                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
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
                f.write(','.join([str(i) for i in x]) + '\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, seq_name, toc, idx / toc))


if __name__ == '__main__':
    main()
