# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import torch
import numpy as np

from external.pysot.pysot import get_axis_aligned_bbox
from external.pysot.toolkit.datasets import DatasetFactory
from external.pysot.toolkit import vot_overlap, vot_float2str

# base tracker
tracker_param_ = 'default'
tracker_name_ = 'eco'
from arena.ar_legacy.VOT2018.common_path import *

from pytracking.evaluation import Tracker
parser = argparse.ArgumentParser(description='Pytracking-RF tracking')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether to visualzie result')
parser.add_argument('--debug', action='store_true',default=False,
        help='whether to debug'),
parser.add_argument('--tracker_name', default= tracker_name_, type=str,
        help='name of tracker for pytracking tracker'),
parser.add_argument('--tracker_param', default= tracker_param_, type=str,
        help='name of param for pytracking tracker')
parser.add_argument('--run_id',type=int, default=1)

args = parser.parse_args()
torch.set_num_threads(1)


def main():
    # create tracker
    tracker_info = Tracker(args.tracker_name, args.tracker_param, None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = args.debug
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    model_name = args.tracker_name + '_' + args.tracker_param

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root_,
                                            load_img=False)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        total_lost = 0

        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            '''对refinement module计时'''
            toc_refine = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                '''get RGB format image'''
                img_RGB = img[:, :, ::-1].copy()  # BGR --> RGB
                if idx == frame_counter:
                    H,W,_ = img.shape
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    '''Initialize'''
                    gt_bbox_np = np.array(gt_bbox_)
                    gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
                    init_info = {}
                    init_info['init_bbox'] = gt_bbox_torch
                    _ = tracker.initialize(img_RGB, init_info)

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    '''Track'''
                    outputs = tracker.track(img_RGB)
                    pred_bbox = outputs['target_bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if len(pred_bbox) == 8:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(save_dir, args.dataset, model_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))


if __name__ == '__main__':
    main()
