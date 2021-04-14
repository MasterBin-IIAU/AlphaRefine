import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import json
import torch
from ltr.admin.environment import env_settings
'''newly added'''
import cv2
from os.path import join
import numpy as np

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys



def get_contour(mask):
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    return contours

class Saliency(BaseDataset):
    """ Saliency dataset.
    """
    def __init__(self, root=None, image_loader=default_image_loader, min_length=0, max_target_area=1):
        """
        args:
            root - path to the imagenet vid dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            min_length - Minimum allowed sequence length.
            max_target_area - max allowed ratio between target area and image area. Can be used to filter out targets
                                which cover complete image.
        """
        root = env_settings().saliency_dir if root is None else root
        super().__init__(root, image_loader)

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                self.sequence_dict = json.load(f)
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_dict = self.get_sequence_dict(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_dict, f)
        # Filter the sequences based on min_length and max_target_area in the first frame
        '''由于在制作saliency数据集时,每帧其实我都已经检查过了,所以这块直接提取name就行了,不用再筛选了'''
        self.sequence_list = [key for key in self.sequence_dict.keys()]

    def get_name(self):
        return 'saliency'
    def is_video_sequence(self):
        return False
    def has_mask(self):
        return True
    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        '''根据seq_id得到被选中的视频的信息,其中visible属性最重要.它被用来判断当前视频是否可以拿来训练'''
        '''需要得到“每一帧是否valid”'''
        bbox_list = [self.sequence_dict['%08d'%(seq_id+1)]]
        bbox_arr = np.array(bbox_list).astype(np.float32) # (N,4)
        bbox = torch.from_numpy(bbox_arr)  # torch tensor (N,4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_id):
        '''已知一个序列,以及待使用的帧在序列中的序号 返回对应的帧以及二值化mask'''
        gt_name = '%08d.png'%(seq_id+1)
        frame_name = '%08d.jpg'%(seq_id+1)
        gt_path = os.path.join(self.root,'gt',gt_name)
        frame_path = os.path.join(self.root,'images',frame_name)
        mask = cv2.imread(gt_path,0)
        frame = self.image_loader(frame_path)
        mask_img = mask[...,np.newaxis] # (H,W,1)
        mask_ins = (mask_img == 255).astype(np.uint8) # binary mask # (H,W,1)
        '''返回一个元组,第一个元素是RGB格式的图像,第二个元素是单通道的mask(只有0,1两种取值,但是是uint8类型)'''
        return (frame, mask_ins)

    def get_frames(self, seq_id, frame_ids, anno=None):

        frame_mask_list = [self._get_frame(seq_id) for f in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...].clone() for _ in frame_ids]
        '''return both frame and mask'''
        frame_list = [f for f,m in frame_mask_list]
        mask_list = [m for f,m in frame_mask_list]
        return frame_list, mask_list, anno_frames, None


    def get_sequence_dict(self, data_dir):
        gt_dir = os.path.join(data_dir,'gt')
        img_dir = os.path.join(data_dir,'images')
        gt_list = sorted(os.listdir(gt_dir))
        sequence_list = [gt_name.replace('.png','') for gt_name in gt_list]
        '''用字典把每张图片的边界框保存下来'''
        sequence_dict = {}
        for idx, name in enumerate(sequence_list):
            gt_path = os.path.join(gt_dir,'%s.png'%name)
            gt_mask = cv2.imread(gt_path,0)
            bbox = xyxy_to_xywh(polys_to_boxes(get_contour(gt_mask))).tolist()[0]
            sequence_dict[name] = bbox # list
        return sequence_dict


