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
from pysot.toolkit.datasets import DatasetFactory
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip

# base tracker
from arena.LaSOT.common_path import *
import random


###################################################
'''dimp'''
# tracker_param_ = 'dimp50_RF'
# tracker_param_ = 'dimp50_vot_RF'
# tracker_param_ = 'dimp50_vot'
# tracker_param_ = 'dimp50'
tracker_param_ = 'super_dimp'
###################################################
'''dimp'''
tracker_name_ = 'dimp'

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


COLORS = [
    (72.8571, 255.0000, 0),     # gt
    (255.0000, 218.5714, 0),  # base tracker
    (0, 145.7143, 255.0000),  # iou-net
    (0, 255.0000, 145.7143),    # siammask
    (255.0000, 0, 0),         # ar
    (72.8571, 0, 255.0000),
    (255.0000, 0, 218.5714),
]


vis_list = ['dog-19', 'motorcycle-1', 'rubicCube-14', 'surfboard-8',
            'crocodile-3', 'kangaroo-5', 'motorcycle-18', 'rubicCube-19', 'tank-9',
            'dog-1', 'lion-20', 'rabbit-17', 'sheep-9', 'turtle-16',
            'dog-15', 'monkey-4', 'rubicCube-1', 'squirrel-8', 'umbrella-9']


def main():
    # create tracker
    tracker_info = Tracker(args.tracker_name, args.tracker_param, None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = args.debug
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    '''Refinement module'''
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)
    model_name = args.tracker_name + '_' + args.tracker_param + '{}-{}'.format(RF_type, selector_path) + '_%d'%(args.run_id)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root_, load_img=False)

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        color = np.array(COLORS[random.randint(0, len(COLORS) - 1)])[None, None, ::-1]
        vis_result = os.path.join('/home/zxy/Desktop/AlphaRefine/CVPR21/material/quality_analysis/mask_vis', '{}'.format(video.name))

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
            else:
                print()

        if not os.path.exists(vis_result):
            os.makedirs(vis_result)


        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            '''get RGB format image'''
            img_RGB = img[:, :, ::-1].copy()  # BGR --> RGB
            tic = cv2.getTickCount()
            if idx == 0:
                H, W, _ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
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
                pred_bbox = RF_module.refine(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.array(pred_bbox))

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

                mask_pred = RF_module.get_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.array(pred_bbox))
                from pysot.toolkit.visualization.draw_mask import draw_mask
                draw_mask(img, mask_pred, idx=idx, show=True, save_dir='dimpsuper_armask_crocodile-3')

                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                im4show = img
                mask_pred = np.uint8(mask_pred > 0.5)[:, :, None]
                contours, _ = cv2.findContours(mask_pred.squeeze(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                im4show = im4show * (1 - mask_pred) + np.uint8(im4show * mask_pred/2) + mask_pred * np.uint8(color) * 128
                pred_bbox = list(map(int, pred_bbox))
                # gt_bbox = list(map(int, gt_bbox))
                # cv2.rectangle(im4show, (gt_bbox[0], gt_bbox[1]),
                #               (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)

                # cv2.rectangle(im4show, (pred_bbox[0], pred_bbox[1]),
                #               (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), color[::-1].squeeze().tolist(), 3)

                cv2.drawContours(im4show, contours, -1, color[::-1].squeeze(), 2)
                cv2.putText(im4show, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # cv2.imshow(video.name, im4show)
                cv2.imwrite(os.path.join(vis_result, '{:06}.jpg'.format(idx)), im4show)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        # model_path = os.path.join(save_dir, args.dataset, model_name)
        # if not os.path.isdir(model_path):
        #     os.makedirs(model_path)
        # result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        # with open(result_path, 'w') as f:
        #     for x in pred_bboxes:
        #         f.write(','.join([str(i) for i in x])+'\n')
        # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        #     v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
