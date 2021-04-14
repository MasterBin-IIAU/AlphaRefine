import torch
import numpy as np
import cv2


def add_frame_mask(frame, mask, threshold=0.5):
    mask_new = (mask > threshold) * 255  # (H,W)
    frame_new = frame.copy().astype(np.float)
    frame_new[..., 1] += 0.3 * mask_new
    frame_new = frame_new.clip(0, 255).astype(np.uint8)
    return frame_new


def add_frame_bbox(frame, refined_box, color):
    x1, y1, w, h = refined_box.tolist()
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    return frame


def mask2bbox(mask, ori_bbox, MASK_THRESHOLD=0.5):
    target_mask = (mask > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boundingRect(polygon)
    else:  # empty mask
        prbox = ori_bbox
    return np.array(prbox).astype(np.float)


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh
