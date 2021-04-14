

def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
    '''boundary (H,W)'''
    x1_new = max(0, min(x1, boundary[1] - min_sz))
    y1_new = max(0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    return x1_new, y1_new, x2_new, y2_new