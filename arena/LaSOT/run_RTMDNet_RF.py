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

from arena.LaSOT.common_path import *
from external.RT_MDNet.MDNet_utils import bbox_clip

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
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root_, load_img=False)

    '''##### build a Refinement module #####'''
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)
    model_name = 'RTMDNet' + '{}-{}'.format(RF_type, selector_path)

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if os.path.exists(os.path.join(save_dir, args.dataset, model_name, '{}.txt'.format(video.name))):
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        '''build tracker'''
        tracker = RT_MDNet()
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # RGB format
            tic = cv2.getTickCount()
            if idx == 0:
                H,W,_ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                '''initialize tracker'''
                tracker.initialize_seq(img_RGB, np.array(gt_bbox_))
                '''initilize refine module for specific video'''
                RF_module.initialize(img_RGB, np.array(gt_bbox_))
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                ori_bbox = tracker.track(img_RGB)
                '''##### refine tracking results #####'''
                pred_bbox = RF_module.refine(img_RGB, np.array(ori_bbox))
                '''boundary and size limit'''
                pred_bbox = bbox_clip(pred_bbox,(H,W))
                '''update state'''
                tracker.target_bbox = pred_bbox.copy()
                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                ori_bbox = list(map(int,ori_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 0, 255), 3)
                cv2.rectangle(img, (ori_bbox[0], ori_bbox[1]),
                              (ori_bbox[0] + ori_bbox[2], ori_bbox[1] + ori_bbox[3]), (255, 0, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 0), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()


        # save results
        model_path = os.path.join(save_dir, args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
