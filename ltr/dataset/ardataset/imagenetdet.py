import os
import cv2

from .base_dataset import BaseDataset
import xml.etree.ElementTree as ET
import json
import torch
from ltr.admin.environment import env_settings

import glob
from os.path import join


def get_target_to_image_ratio(seq):
    anno = torch.Tensor(seq['anno'])
    img_sz = torch.Tensor(seq['image_size'])
    return (anno[:4].prod() / (img_sz.prod())).sqrt()


def default_image_loader(path):
    def opencv_loader(path):
        """ Read image using opencv's imread function and returns it in rgb format"""
        try:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            # convert to rgb and return
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None
    return opencv_loader(path)


class ImagenetDET(BaseDataset):
    """ Imagenet DET dataset.

    Publication:
        ImageNet Large Scale Visual Recognition Challenge
        Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
        Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei
        IJCV, 2015
        https://arxiv.org/pdf/1409.0575.pdf

    Download the dataset from http://image-net.org/
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
        root = env_settings().imagenetdet_dir if root is None else root
        super().__init__(root, image_loader)

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_list = self.get_seqence_list(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)

        # Filter the sequences based on min_length and max_target_area in the first frame
        self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]
        self.dataset_root = root

    def get_name(self):
        return 'imagenetdet'

    def is_video_sequence(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        """ 根据seq_id得到被选中的视频的信息,其中visible属性最重要.它被用来判断当前视频是否可以拿来训练 """
        bbox = torch.Tensor(self.sequence_list[seq_id]['anno']).view(1, 4)  # torch tensor (1,4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_id):
        relative_path = self.sequence_list[seq_id]['path']
        frame_path = join(self.dataset_root, 'Data/DET/train/', relative_path)
        return self.image_loader(frame_path)

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frame(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids] # anno['bbox']--->(4,) torch tensor

        # object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, None

    def get_seqence_list(self, root):
        sequence_list = []
        ann_base_path = join(root, 'Annotations/DET/train/')
        sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h') # without i
        for sub_set in sub_sets:
            sub_set_base_path = join(ann_base_path, sub_set)
            if 'a' == sub_set:
                xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
            else:
                xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
            for xml in xmls:
                Cimg_anno = ET.parse(xml)
                image_size = [int(Cimg_anno.find('size/width').text), int(Cimg_anno.find('size/height').text)] # (w,h)
                objects = Cimg_anno.findall('object')
                for idx, obj in enumerate(objects):
                    bndbox = obj.find('bndbox')
                    x1 = int(bndbox.find('xmin').text)
                    y1 = int(bndbox.find('ymin').text)
                    x2 = int(bndbox.find('xmax').text)
                    y2 = int(bndbox.find('ymax').text)
                    bbox = [x1, y1, x2-x1, y2-y1]#(x1,y1,w,h)
                    class_name_id = obj.find('name').text
                    relative_path = xml.replace(sub_set_base_path,sub_set).replace('.xml','.JPEG')
                    new_sequence = {'path':relative_path, 'anno': bbox,
                                    'image_size': image_size, 'class_name': class_name_id}
                    sequence_list.append(new_sequence)
        return sequence_list

