# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import torch
import numpy as np

from external.pysot.pysot.utils.bbox import get_axis_aligned_bbox
from external.pysot.toolkit.datasets import DatasetFactory
from external.pysot.toolkit.utils.region import vot_overlap, vot_float2str
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip

# base tracker
from arena.LaSOT.common_path_siamrpn import *


dataset_name_ = 'LaSOT'
# dataset_name_ = 'VOT2018'
# dataset_name_ = 'OTB100'
###################################################
video_name_ = ''
# video_name_ = 'airplane-9'
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
parser.add_argument('--gpu_id', default=1, type=int)


args = parser.parse_args()

torch.set_num_threads(1)
torch.cuda.set_device(args.gpu_id) # set GPU id


def main():
    # load config
    dataset_root = '/media/zxy/Samsung_T5/Data/DataSets/LaSOT/LaSOT_test'

    # create tracker
    '''Pytracking-RF tracker'''
    tracker_info = Tracker(args.tracker_name, args.tracker_param, None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = args.debug
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    '''Refinement module'''

    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm-8gpu/SEbcmnet_ep0040.pth.tar"  # RF_CrsM_ARv1_d
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEbcm/SEbcm-8gpu/SEbcm_org/SEbcmnet_ep0040.pth-a.tar"  # RF_AR_org_8gpu_a
    # selector_path = 1; branches = ['corner', 'mask'][0:1]
    # sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default


    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_15sr_fcn/SEcmnet_ep0040-a.pth.tar"  # RF_CrsM_R34SR15FCN_a

    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-a.pth.tar"  # dimp_dimp50RF_CrsM_R34SR20FCN_a-0_1
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-b.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-d.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34/SEcmnet_ep0040-e.pth.tar"



    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-a.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-b.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-c.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-d.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcrsm_rmNL/SEcmnet_ep0040-e.pth.tar"  # RF_CrsM_R50SR20_e

    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEcm/SEcm/SEcmnet_ep0040.pth-d.tar"

    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-a.pth.tar"
    refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-b.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-c.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-d.pth.tar"
    # refine_path = "/home/zxy/Desktop/AlphaRefine/experiments/SEx_beta/SEcm_r34_woPr/SEcmnet_ep0040-e.pth.tar"

    selector_path = 0; branches = ['corner', 'mask'][0:1]
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default

    RF_module = RefineModule(refine_path, selector_path, branches=branches, search_factor=sr, input_sz=input_sz)

    RF_type = 'RF_CrsM_woPr_R34SR20_b'

    model_name = args.tracker_name + '_' + args.tracker_param + '{}-{}'.format(RF_type, selector_path) + '_%d'%(args.run_id)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
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
                    '''##### initilize refinement module for specific video'''
                    RF_module.initialize(img_RGB, np.array(gt_bbox_))

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    '''Track'''
                    outputs = tracker.track(img_RGB)
                    pred_bbox = outputs['target_bbox']
                    '''##### refine tracking results #####'''
                    result_dict = RF_module.refine(img_RGB, np.array(pred_bbox))
                    bbox_report = result_dict['bbox_report']
                    bbox_state = result_dict['bbox_state']
                    '''report result and update state'''
                    pred_bbox = bbox_report
                    x1, y1, w, h = bbox_state.tolist()
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
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))

    else:
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
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
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
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(save_dir, args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(save_dir, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
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
