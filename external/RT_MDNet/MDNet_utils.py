
import numpy as np
def bbox_clip(bbox, boundary, min_sz=10.0):
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    x1_new = max(0.0, min(x1, boundary[1] - min_sz))
    y1_new = max(0.0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    w_new = x2_new - x1_new
    h_new = y2_new - y1_new
    '''get new bbox'''
    bbox_new = np.array([x1_new, y1_new, w_new, h_new])
    return bbox_new