import argparse
import cv2

import torch
from arena.VOT2020.utils import make_full_size, VOT
from arena.VOT2020.trackers.pytracking_ar import BaseTracker


def build_tracker(tracker_name, para_name, threshold):
    tracker = BaseTracker(tracker_name=tracker_name, para_name=para_name, threshold=threshold)
    return tracker


def run_vot_exp(tracker_name, para_name, threshold):
    tracker = build_tracker(tracker_name, para_name, threshold)

    handle = VOT("mask")
    selection = handle.region()
    imagefile = handle.frame()

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, m = tracker.track(image)
        handle.report(m)


def parse_args():
    from arena.VOT2020.common_path import _tracker_name, _tracker_param, _mask_thres
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", default=_tracker_name, help="tracker_name options: dimp|atom|eco")
    parser.add_argument("--tracker_param", default=_tracker_param,
                        help="tracker_param options: super_dimp;dimp50|default|default")
    parser.add_argument("--mask_thres", type=float, default=_mask_thres,
                        help="tracker_param options: super_dimp;dimp50|default|default")
    args = parser.parse_args()
    return args


args = parse_args()
torch.set_num_threads(1)
print(args)
run_vot_exp(args.tracker_name, args.tracker_param, args.mask_thres)
