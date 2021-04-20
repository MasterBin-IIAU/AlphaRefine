# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import torch
import numpy as np
'''dataset'''
from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox

'''Tracker module'''
from external.RT_MDNet.RT_MDNet import RT_MDNet
'''Refine module'''

from arena.TrackingNet.common_path import *

parser = argparse.ArgumentParser(description='RT-MDNet Refine tracking')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result'),
args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # create dataset
    dataset_root = dataset_root_
    frames_dir = os.path.join(dataset_root, 'frames')
    seq_list = sorted(os.listdir(frames_dir))

    model_name = 'RTMDNet'

    # OPE tracking
    for v_idx, seq_name in enumerate(seq_list):
        if args.video != '':
            # test one special video
            if seq_name != args.video:
                continue
        '''build tracker'''
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        tracker = RT_MDNet()
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
                H,W,_ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                '''initialize tracker'''
                tracker.initialize_seq(img_RGB, np.array(gt_bbox_))
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)

            else:
                pred_bbox = tracker.track(img_RGB)
                pred_bboxes.append(pred_bbox)

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
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
