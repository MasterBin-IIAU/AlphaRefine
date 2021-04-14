import os
import sys
import cv2

import numpy as np
import torch
import vot as vot
from arena.VOT2020.utils import make_full_size


def build_tracker(tracker_name, para_name, refine_model_name, threshold):
    tracker = DIMP_ALPHA(tracker_name=tracker_name,para_name=para_name,
                         refine_model_name=refine_model_name,threshold=threshold)
    return tracker, None


def run_vot_exp(tracker_name,para_name,refine_model_name,threshold,VIS=False):

    torch.set_num_threads(1)
    save_root = os.path.join('/home/alphabin/Desktop/AlphaRefine_submit/vot20_debug',para_name)
    if VIS and (not os.path.exists(save_root)):
        os.mkdir(save_root)

    tracker = build_tracker(tracker_name, para_name, refine_model_name, threshold)

    handle = vot.VOT("mask")
    selection = handle.region()
    imagefile = handle.frame()
    # if not imagefile:
    #     sys.exit(0)

    # if VIS:
    #     '''for vis'''
    #     seq_name = imagefile.split('/')[-3]
    #     save_v_dir = os.path.join(save_root,seq_name)
    #     if not os.path.exists(save_v_dir):
    #         os.mkdir(save_v_dir)
    #     cur_time = int(time.time() % 10000)
    #     save_dir = os.path.join(save_v_dir, str(cur_time))
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, m, search, search_m = tracker.track(image)
        handle.report(m)


        # if VIS:
        #     '''Visualization'''
        #     # original image
        #     image_ori = image[:,:,::-1].copy() # RGB --> BGR
        #     image_name = imagefile.split('/')[-1]
        #     save_path = os.path.join(save_dir, image_name)
        #     cv2.imwrite(save_path, image_ori)
        #     # dimp box
        #     image_b = image_ori.copy()
        #     cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
        #                   (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
        #     image_b_name = image_name.replace('.jpg','_bbox.jpg')
        #     save_path = os.path.join(save_dir, image_b_name)
        #     cv2.imwrite(save_path, image_b)
        #     # search region
        #     search_bgr = search[:,:,::-1].copy()
        #     search_name = image_name.replace('.jpg', '_search.jpg')
        #     save_path = os.path.join(save_dir, search_name)
        #     cv2.imwrite(save_path, search_bgr)
        #     # search region mask
        #     search_bgr_m = search_bgr.astype(np.float32)
        #     search_bgr_m[:, :, 1] += 127.0 * search_m
        #     search_bgr_m[:, :, 2] += 127.0 * search_m
        #     contours, _ = cv2.findContours(search_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     search_bgr_m = cv2.drawContours(search_bgr_m, contours, -1, (0, 255, 255), 4)
        #     search_bgr_m = search_bgr_m.clip(0,255).astype(np.uint8)
        #     search_name_m = image_name.replace('.jpg', '_search_mask.jpg')
        #     save_path = os.path.join(save_dir, search_name_m)
        #     cv2.imwrite(save_path, search_bgr_m)
        #     # original image + mask
        #     image_m = image_ori.copy().astype(np.float32)
        #     image_m[:, :, 1] += 127.0 * m
        #     image_m[:, :, 2] += 127.0 * m
        #     contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     image_m = cv2.drawContours(image_m, contours, -1, (0, 255, 255), 2)
        #     image_m = image_m.clip(0, 255).astype(np.uint8)
        #     image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
        #     save_path = os.path.join(save_dir, image_mask_name_m)
        #     cv2.imwrite(save_path, image_m)
