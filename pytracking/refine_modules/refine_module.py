import time
import torch
import numpy as np
import cv2

from pytracking.utils.loading import load_network
from ltr.data.processing_utils_SE import sample_target_SE, transform_image_to_crop_SE, map_mask_back
from .utils import mask2bbox, delta2bbox


class RefineModule(object):
    def __init__(self, refine_net_dir, selector=None, search_factor=2.0, input_sz=256):
        self.refine_network = self.get_network(refine_net_dir)
        assert isinstance(selector, (int, str))
        self.branch_selector = selector if isinstance(selector, int) else self.get_network(selector)
        self.search_factor = search_factor
        self.input_sz = input_sz
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def initialize(self, frame1, bbox1):
        """
        Args:
            frame1(np.array): cv2 iamge array with shape (H,W,3)
            bbox1(np.array): with shape(4,)
        """

        """Step1: get cropped patch(tensor)"""
        patch1, h_f, w_f = sample_target_SE(frame1, bbox1, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        patch1_tensor = self.img_preprocess(patch1)

        """Step2: get GT's cooridinate on the cropped patch(tensor)"""
        crop_sz = torch.Tensor((self.input_sz, self.input_sz))
        bbox1_tensor = self.gt_preprocess(bbox1)  # (4,)
        bbox1_crop_tensor = transform_image_to_crop_SE(bbox1_tensor, bbox1_tensor, h_f, w_f, crop_sz).cuda()

        """Step3: forward prop (reference branch)"""
        with torch.no_grad():
            self.refine_network.forward_ref(patch1_tensor, bbox1_crop_tensor)

    def refine(self, Cframe, Cbbox, mode='all', test=False):
        """
        Args:
            Cframe: Current frame(cv2 array)
            Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        """
        tic = time.time()
        if mode not in ['bbox', 'mask', 'corner', 'all']:
            raise ValueError("mode should be 'bbox' or 'mask' or 'corner' or 'all' ")

        """ Step1: get cropped patch (search region) """
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        Cpatch_tensor = self.img_preprocess(Cpatch)

        """ Step2: forward prop (test branch) """
        output_dict = {}
        with torch.no_grad():

            output = self.refine_network.forward_test(Cpatch_tensor, mode='test')  # (1,1,H,W)

            if mode == 'bbox' or mode == 'corner':
                Pbbox_arr = self.pred2bbox(output, input_type=mode)
                output_dict[mode] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)

            elif mode == 'mask':
                Pmask_arr = self.pred2bbox(output, input_type=mode)
                output_dict['mask'] = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                                    mode=cv2.BORDER_CONSTANT)

            else:
                boxes = []
                box = [0, 0, 0, 0]
                if 'bbox' in output:
                    Pbbox_arr = self.pred2bbox(output, input_type='bbox')
                    output_dict['bbox'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
                    boxes.append(output_dict['bbox'])
                    box = output_dict['bbox']  # for mask absense

                if 'corner' in output:
                    Pbbox_arr = self.pred2bbox(output, input_type='corner')
                    output_dict['corner'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
                    boxes.append(output_dict['corner'])
                    box = output_dict['corner']

                if 'mask' in output:
                    Pmask_arr = self.pred2bbox(output, input_type='mask')
                    output_dict['mask'] = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                                        mode=cv2.BORDER_CONSTANT)
                    output_dict['mask_bbox'] = mask2bbox(output_dict['mask'], box)
                    boxes.append(output_dict['mask_bbox'])

                if not isinstance(self.branch_selector, int):
                    branch_scores = self.branch_selector(output['feat'])
                    _, max_idx = torch.max(branch_scores.squeeze(), dim=0)
                    max_idx = max_idx.item()
                else:
                    max_idx = self.branch_selector
                output_dict['all'] = boxes[max_idx]
        # print(time.time() - tic)
        return output_dict if test else output_dict[mode]

    def get_mask(self, Cframe, Cbbox, mode='all', test=False):
        """
        Args:
            Cframe: Current frame(cv2 array)
            Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        """
        if mode not in ['bbox', 'mask', 'corner', 'all']:
            raise ValueError("mode should be 'bbox' or 'mask' or 'corner' or 'all' ")

        """ Step1: get cropped patch (search region) """
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        Cpatch_tensor = self.img_preprocess(Cpatch)

        """ Step2: forward prop (test branch) """
        output_dict = {}
        with torch.no_grad():

            output = self.refine_network.forward_test(Cpatch_tensor, mode='test', branches=['mask'])  # (1,1,H,W)

            assert 'mask' in output
            Pmask_arr = self.pred2bbox(output, input_type='mask')
            output_dict['mask'] = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                                mode=cv2.BORDER_CONSTANT)

        return output_dict['mask']

    def pred2bbox(self, prediction, input_type=None):
        if input_type == 'bbox':
            Pbbox = prediction['bbox']
            Pbbox = delta2bbox(Pbbox)
            Pbbox_arr = np.array(Pbbox.squeeze().cpu())
            return Pbbox_arr

        elif input_type == 'corner':
            Pcorner = prediction['corner']  # (x1,y1,x2,y2)
            Pbbox_arr = np.array(Pcorner.squeeze().cpu())
            Pbbox_arr[2:] = Pbbox_arr[2:] - Pbbox_arr[:2]  # (x1,y1,w,h)
            return Pbbox_arr

        elif input_type == 'mask':
            Pmask = prediction['mask']
            Pmask_arr = np.array(Pmask.squeeze().cpu())  # (H,W) (0,1)
            return Pmask_arr

        else:
            raise ValueError("input_type should be 'bbox' or 'mask' or 'corner' ")

    def bbox_back(self, bbox_crop, bbox_ori, h_f, w_f):
        """
        Args:
            bbox_crop: coordinate on (256x256) region in format (x1,y1,w,h) (4,)
            bbox_ori: origin traking result (x1,y1,w,h) (4,)
            h_f: h scale factor
            w_f: w scale factor
        Return:
            coordinate mapping back to origin image
        """
        x1_c, y1_c, w_c, h_c = bbox_crop.tolist()
        x1_o, y1_o, w_o, h_o = bbox_ori.tolist()
        x1_oo = x1_o - (self.search_factor-1)/2 * w_o
        y1_oo = y1_o - (self.search_factor-1)/2 * h_o
        delta_x1 = x1_c / w_f
        delta_y1 = y1_c / h_f
        delta_w = w_c / w_f
        delta_h = h_c / h_f
        return np.array([x1_oo + delta_x1, y1_oo + delta_y1,
                         delta_w, delta_h])

    def get_network(self, checkpoint_dir):
        network = load_network(checkpoint_dir)
        network.cuda()
        network.eval()
        return network

    def img_preprocess(self, img_arr):
        """ to torch.Tensor(RGB), normalized (minus mean, divided by std)
        Args:
            img_arr: (H,W,3)
        Return:
            (1,1,3,H,W)
        """
        norm_img = ((img_arr / 255.0) - self.mean) / (self.std)
        img_f32 = norm_img.astype(np.float32)
        img_tensor = torch.from_numpy(img_f32).cuda()
        img_tensor = img_tensor.permute((2, 0, 1))
        return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    def gt_preprocess(self, gt_arr):
        """
        Args:
            gt_arr: ndarray (4,)
        Return:
            `torch.Tensor` (4,)
        """
        return torch.from_numpy(gt_arr.astype(np.float32))

