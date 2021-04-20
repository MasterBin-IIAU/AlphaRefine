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

class Instance(object):
    instID     = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if (instID ==0 ):
            return
        self.instID     = int(instID) # 1
        self.pixelCount = int(self.getInstancePixels(imgNp, instID)) # 目标占据的像素个数

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "("+str(self.instID)+")"

def get_target_to_image_ratio(seq):
    init_frame = seq[0]
    H,W = init_frame['h'],init_frame['w']
    # area = init_frame['area']
    bbox = init_frame['bbox'] # list length=4
    anno = torch.Tensor(bbox)
    img_sz = torch.Tensor([H,W])
    # return (area / (img_sz.prod())).sqrt()
    '''边界框面积与图像面积算比值,再开方'''
    return (anno[2:4].prod() / (img_sz.prod())).sqrt()


class Youtube_VOS(BaseDataset):
    """ Youtube_VOS dataset.
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
        root = env_settings().youtubevos_dir if root is None else root
        super().__init__(root, image_loader)

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the imagenet annotations and generate the cache file
            self.sequence_list = self.convert_ytb_vos(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)
        # Filter the sequences based on min_length and max_target_area in the first frame
        '''youtube-vos中某些视频的目标很大,以至于用边界框来框的话就直接是整张图,我暂时去掉了这些实例(数量不多,感觉影响不大)'''
        self.sequence_list = [x for x in self.sequence_list if len(x) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]
        # print(len(self.sequence_list))
        # for x in self.sequence_list:
        #     if len(x) < min_length:
        #         print('小于最小长度:',x)
        #     if get_target_to_image_ratio(x) >= max_target_area:
        #         print('超出图像范围',x[0]['file_name'])


    def get_name(self):
        return 'youtube_vos'
    def has_mask(self):
        return True
    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        '''根据seq_id得到被选中的视频的信息,其中visible属性最重要.它被用来判断当前视频是否可以拿来训练'''
        '''需要得到“每一帧是否valid”'''
        '''由于数据存在很多字典里,需要转换一下才行'''
        cur_seq = self.sequence_list[seq_id]
        bbox_list = []
        for idx,info_dict in enumerate(cur_seq):
            bbox_list.append(info_dict['bbox'])
        bbox_arr = np.array(bbox_list).astype(np.float32) # (N,4)
        bbox = torch.from_numpy(bbox_arr)  # torch tensor (N,4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, sequence, frame_id):
        '''已知一个序列,以及待使用的帧在序列中的序号 返回对应的帧以及二值化mask'''
        frame_name = sequence[frame_id]['file_name']
        '''image RGB 3 channels'''
        frame_path = join(self.root, 'train/JPEGImages', frame_name + '.jpg')
        frame_img = self.image_loader(frame_path)
        '''mask 1 channel'''
        mask_path = join(self.root,'train/Annotations/',frame_name+'.png')
        mask_img = cv2.imread(mask_path, 0)
        mask_img = mask_img[...,np.newaxis] # (H,W,1)
        mask_ins = (mask_img == sequence[frame_id]['id']).astype(np.uint8) # binary mask # (H,W,1)
        '''返回一个元组,第一个元素是RGB格式的图像,第二个元素是单通道的mask(只有0,1两种取值,但是是uint8类型)'''
        return (frame_img, mask_ins)

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_mask_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        '''return both frame and mask'''
        frame_list = [f for f,m in frame_mask_list]
        mask_list = [m for f,m in frame_mask_list]
        return frame_list, mask_list, anno_frames, None

    def convert_ytb_vos(self, data_dir):
        sets = ['train']
        ann_dirs = ['train/Annotations/']
        num_obj = 0
        num_ann = 0
        for data_set, ann_dir in zip(sets, ann_dirs):
            print('Starting %s' % data_set)
            # ann_dict = {}
            ann_list = []
            ann_dir = os.path.join(data_dir, ann_dir)
            json_ann = json.load(open(os.path.join(ann_dir, '../meta.json')))
            for vid, video in enumerate(json_ann['videos']):
                v = json_ann['videos'][video]
                frames = []
                for obj in v['objects']:
                    o = v['objects'][obj]
                    '''extend方法用于向列表尾部追加多个新元素(相当于合并两个列表)'''
                    frames.extend(o['frames'])
                '''frames记录了当前这个视频里,都有哪些帧出现了目标'''
                frames = sorted(set(frames))

                annotations = []
                instanceIds = []
                '''遍历那些出现(存在)目标的帧'''
                for frame in frames:
                    file_name = join(video, frame)
                    fullname = os.path.join(ann_dir, file_name+'.png')
                    '''注意:这里读取出的img是有(多种)颜色的图像. youtube-vos的标注是这样:
                    每一个目标用一种颜色标注出来'''
                    '''imread()中的第二个参数: 0代表以灰度图像形式读取'''
                    img = cv2.imread(fullname, 0)
                    h, w = img.shape[:2]

                    objects = dict()
                    '''由于每个目标用一种颜色,因此可以用np.unique把目标分离开来'''
                    for instanceId in np.unique(img):
                        if instanceId == 0:
                            continue
                        instanceObj = Instance(img, instanceId)
                        instanceObj_dict = instanceObj.toDict()
                        mask = (img == instanceId).astype(np.uint8)
                        '''提取当前目标的轮廓(cv2 4.1.1和3.x的API略有区别)'''
                        contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        polygons = [c.reshape(-1).tolist() for c in contour]
                        instanceObj_dict['contours'] = [p for p in polygons if len(p) > 4]
                        '''instanceObj_dict中包含以下key: "instID", "pixelCount", "contours" '''
                        if len(instanceObj_dict['contours']) and instanceObj_dict['pixelCount'] > 1000:
                            objects[instanceId] = instanceObj_dict
                        # else:
                        #     cv2.imshow("disappear?", mask)
                        #     cv2.waitKey(0)

                    for objId in objects:
                        if len(objects[objId]) == 0:
                            continue
                        obj = objects[objId]
                        len_p = [len(p) for p in obj['contours']]
                        if min(len_p) <= 4:
                            print('Warning: invalid contours.')
                            continue  # skip non-instance categories

                        ann = dict()
                        ann['h'] = h
                        ann['w'] = w
                        ann['file_name'] = file_name
                        ann['id'] = int(objId)
                        # ann['segmentation'] = obj['contours']
                        # ann['iscrowd'] = 0
                        ann['area'] = obj['pixelCount']
                        ann['bbox'] = xyxy_to_xywh(polys_to_boxes([obj['contours']])).tolist()[0]
                        '''每一帧每一个目标的信息都被添加到了annotations列表中'''
                        annotations.append(ann)
                        '''目标对应的objId被添加到instanceIds列表中'''
                        instanceIds.append(objId)
                        num_ann += 1
                instanceIds = sorted(set(instanceIds))
                num_obj += len(instanceIds)
                '''video_ann是一个字典,其中的key是字符串类型的id,其中的每个value是一个列表,列表中的每个元素是对应帧对应目标的信息字典'''
                video_ann = {str(iId): [] for iId in instanceIds}
                '''把annotations按照实例分离开来'''
                for ann in annotations:
                    video_ann[str(ann['id'])].append(ann)
                '''把分离好的实例级信息(同一实例在一个视频中的轨迹)保存到ann_list中去'''
                for idx in instanceIds:
                    ann_list.append(video_ann[str(idx)])

                if vid % 50 == 0 and vid != 0:
                    print("process: %d video" % (vid+1))

            print("Num Objects: %d" % num_obj)
            print("Num Annotations: %d" % num_ann)

            return ann_list