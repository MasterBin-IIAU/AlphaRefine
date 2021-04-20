# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import torch
import numpy as np
'''dataset'''
from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox
from external.pysot.toolkit.datasets import DatasetFactory
'''Tracker module'''
from external.RT_MDNet.RT_MDNet import RT_MDNet
'''Refine module'''
from pytracking.refine_modules.refine_module import RefineModule

from arena.ar_legacy.VOT2018.common_path import *
from external.RT_MDNet.MDNet_utils import bbox_clip

parser = argparse.ArgumentParser(description='RT-MDNet Refine tracking')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--run_id',type=int, default=1)

args = parser.parse_args()

torch.set_num_threads(1)


def main(model_code):
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root_, load_img=False)

    '''##### build a Refinement module #####'''
    RF_module = RefineModule(refine_path.format(model_code), selector_path, search_factor=sr,
                             input_sz=input_sz)
    model_name = 'RT_MDNet' + '_{}-{}'.format(RF_type.format(model_code), selector_path) + '_%d' % (args.run_id)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            tracker = RT_MDNet()
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # RGB format
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    H,W,_ = img.shape
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    '''initialize tracker'''
                    tracker.initialize_seq(img_RGB, np.array(gt_bbox_))
                    '''initilize refine module for specific video'''
                    RF_module.initialize(img_RGB, np.array(gt_bbox_))
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    '''track'''
                    ori_bbox = tracker.track(img_RGB)
                    '''refine tracking result'''
                    pred_bbox = RF_module.refine(img_RGB, np.array(ori_bbox))
                    pred_bbox = bbox_clip(pred_bbox, (H, W))
                    tracker.target_bbox = pred_bbox.copy()
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
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()

            # save results
            video_path = os.path.join(save_dir, args.dataset, model_name,
                    'baseline', video.name)
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


if __name__ == '__main__':
    for model_code in ['a', 'b', 'c', 'd', 'e']:
        main(model_code)
