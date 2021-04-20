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
from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox
from external.pysot.pysot.utils.model_load import load_pretrain
from external.pysot.toolkit.datasets import DatasetFactory

from pytracking.refine_modules.refine_module import RefineModule

'''RF utils'''
from pytracking.RF_utils import bbox_clip
'''common paths'''
from arena.LaSOT.common_path_siamrpn import *


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str,
                    help='siamrpn_r50_l234_dwxcorr, siamrpn_r50_l234_dwxcorr_otb')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
# parser.add_argument('--refine_method',default='RF',type=str,help='RF, iou_net, mask')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')

args = parser.parse_args()
torch.set_num_threads(1)
n_iter = 1

def main():
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)

    # refine_method = args.refine_method
    model_name = 'siamrpn_' + RF_type + '-iter{}'.format(n_iter)

    snapshot_path = os.path.join(project_path_, 'experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'experiments/%s/config.yaml' % args.tracker_name)

    cfg.merge_from_file(config_path)

    # create model
    model = ModelBuilder()  # a sub-class of `torch.nn.Module`
    model = load_pretrain(model, snapshot_path).cuda().eval()

    # build tracker
    tracker = build_tracker(model)  # tracker is a object consisting of NN and some post-processing

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root_, load_img=False)

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if os.path.exists(os.path.join(save_dir, args.dataset, model_name, '{}.txt'.format(video.name))):
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                H,W,_ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                '''##### initilize refinement module for specific video'''
                RF_module.initialize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.array(gt_bbox_))
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)

            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                '''##### refine tracking results #####'''
                for i in range(n_iter):
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
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        model_path = os.path.join(save_dir, args.dataset, model_name+'_'+str(selector_path))
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
