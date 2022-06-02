import argparse
import distutils
import os
import glob
import torch
import numpy as np
import sys
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip
import time
from pathlib import Path
import math
import copy
import torch.nn.functional as F
import torch.nn as nn
import cv2

#pytracking/RF_utils
def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
    '''boundary (H,W)'''
    x1_new = max(0, min(x1, boundary[1] - min_sz))
    y1_new = max(0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    return x1_new, y1_new, x2_new, y2_new
####

#pytracking/refine_modules/utils
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
####

#pytracking/refine_modules/refine_module
class RefineModule(object):
    def __init__(self, refine_net_dir, selector=None, search_factor=2.0, input_sz=256):
        #print('refine_net_dir = ' + refine_net_dir)
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
        #print('refine network load')
        #print('checkpoint_dir = ' + checkpoint_dir)
        network, _ = load_network(checkpoint_dir)
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
####

#ltr/data/processing_utils_SE
def sample_target_SE(im, target_bb, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, 1.0, 1.0
'''把mask映射到原图上'''
def map_mask_back(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    H,W = (im.shape[0],im.shape[1])
    base = np.zeros((H,W))
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    '''pad base'''
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    '''Resize mask'''
    mask_rsz = cv2.resize(mask,(ws,hs))
    '''fill region with mask'''
    base_padded[y1+y1_pad:y2+y1_pad, x1+x1_pad:x2+x1_pad] = mask_rsz.copy()
    '''crop base_padded to get final mask'''
    final_mask = base_padded[y1_pad:y1_pad+H,x1_pad:x1_pad+W]
    assert (final_mask.shape == (H,W))
    return final_mask

'''Added on 2019.12.23'''
def transform_image_to_crop_SE(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor_h: float, resize_factor_w: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5*box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5*box_in[2:4]

    box_out_xc = (crop_sz[0] -1)/2 + (box_in_center[0] - box_extract_center[0])*resize_factor_w
    box_out_yc = (crop_sz[0] -1)/2 + (box_in_center[1] - box_extract_center[1])*resize_factor_h
    box_out_w = box_in[2] * resize_factor_w
    box_out_h = box_in[3] * resize_factor_h

    '''2019.12.28 为了避免出现(x1,y1)小于0,或者(x2,y2)大于256的情况,这里我对它们加上了一些限制'''
    max_sz = crop_sz[0].item()
    box_out_x1 = torch.clamp(box_out_xc - 0.5 * box_out_w,0,max_sz)
    box_out_y1 = torch.clamp(box_out_yc - 0.5 * box_out_h,0,max_sz)
    box_out_x2 = torch.clamp(box_out_xc + 0.5 * box_out_w,0,max_sz)
    box_out_y2 = torch.clamp(box_out_yc + 0.5 * box_out_h,0,max_sz)
    box_out_w_new = box_out_x2 - box_out_x1
    box_out_h_new = box_out_y2 - box_out_y1
    box_out = torch.stack((box_out_x1, box_out_y1, box_out_w_new, box_out_h_new))
    return box_out
####

#bounding_box_utils
def rect_to_rel(bb, sz_norm=None):
    """Convert standard rectangular parametrization of the bounding box [x, y, w, h]
    to relative parametrization [cx/sw, cy/sh, log(w), log(h)], where [cx, cy] is the center coordinate.
    args:
        bb  -  N x 4 tensor of boxes.
        sz_norm  -  [N] x 2 tensor of value of [sw, sh] (optional). sw=w and sh=h if not given.
    """

    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)

def rel_to_rect(bb, sz_norm=None):
    """Inverts the effect of rect_to_rel. See above."""

    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)
#####

#ltr/models/target_classifier/initializer
class FilterInitializerZero(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, feature_dim=256):
        super().__init__()

        self.filter_size = (feature_dim, filter_size, filter_size)

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""

        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        return feat.new_zeros(num_sequences, self.filter_size[0], self.filter_size[1], self.filter_size[2])
####

#dcf module
def max2d(a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax
def hann1d(sz: int, centered = True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
    return torch.cat([w, w[1:sz-sz//2].flip((0,))])
def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)
def hann2d_clipped(sz: torch.Tensor, effective_sz: torch.Tensor, centered = True) -> torch.Tensor:
    """1D clipped cosine window."""

    # Ensure that the difference is even
    effective_sz += (effective_sz - sz) % 2
    effective_window = hann1d(effective_sz[0].item(), True).reshape(1, 1, -1, 1) * hann1d(effective_sz[1].item(), True).reshape(1, 1, 1, -1)

    pad = (sz - effective_sz) / 2

    window = F.pad(effective_window, (pad[1].item(), pad[1].item(), pad[0].item(), pad[0].item()), 'replicate')

    if centered:
        return window
    else:
        mid = (sz / 2).int()
        window_shift_lr = torch.cat((window[:, :, :, mid[1]:], window[:, :, :, :mid[1]]), 3)
        return torch.cat((window_shift_lr[:, :, mid[0]:, :], window_shift_lr[:, :, :mid[0], :]), 2)
####

#pytracking/utils/params
class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)
####

#pytracking/features/augmentation
class Transform:
    """Base data augmentation transform class."""

    def __init__(self, output_sz = None, shift = None):
        self.output_sz = output_sz
        self.shift = (0,0) if shift is None else shift

    def __call__(self, image, is_mask=False):
        raise NotImplementedError

    def crop_to_output(self, image):
        if isinstance(image, torch.Tensor):
            imsz = image.shape[2:]
            if self.output_sz is None:
                pad_h = 0
                pad_w = 0
            else:
                pad_h = (self.output_sz[0] - imsz[0]) / 2
                pad_w = (self.output_sz[1] - imsz[1]) / 2

            pad_left = math.floor(pad_w) + self.shift[1]
            pad_right = math.ceil(pad_w) - self.shift[1]
            pad_top = math.floor(pad_h) + self.shift[0]
            pad_bottom = math.ceil(pad_h) - self.shift[0]

            return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 'replicate')
        else:
            raise NotImplementedError

class Identity(Transform):
    """Identity transformation."""
    def __call__(self, image, is_mask=False):
        return self.crop_to_output(image)

class FlipHorizontal(Transform):
    """Flip along horizontal axis."""
    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((3,)))
        else:
            return np.fliplr(image)

class FlipVertical(Transform):
    """Flip along vertical axis."""

    def __call__(self, image: torch.Tensor, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((2,)))
        else:
            return np.flipud(image)

class Translation(Transform):
    """Translate."""
    def __init__(self, translation, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.shift = (self.shift[0] + translation[0], self.shift[1] + translation[1])

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image)
        else:
            raise NotImplementedError

class Scale(Transform):
    """Scale."""
    def __init__(self, scale_factor, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.scale_factor = scale_factor

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            # Calculate new size. Ensure that it is even so that crop/pad becomes easier
            h_orig, w_orig = image.shape[2:]

            if h_orig != w_orig:
                raise NotImplementedError

            h_new = round(h_orig /self.scale_factor)
            h_new += (h_new - h_orig) % 2
            w_new = round(w_orig /self.scale_factor)
            w_new += (w_new - w_orig) % 2

            image_resized = F.interpolate(image, [h_new, w_new], mode='bilinear')

            return self.crop_to_output(image_resized)
        else:
            raise NotImplementedError

class Rotate(Transform):
    """Rotate with given angle."""
    def __init__(self, angle, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.angle = math.pi * angle/180

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(numpy_to_torch(self(torch_to_numpy(image))))
        else:
            c = (np.expand_dims(np.array(image.shape[:2]),1)-1)/2
            R = np.array([[math.cos(self.angle), math.sin(self.angle)],
                          [-math.sin(self.angle), math.cos(self.angle)]])
            H =np.concatenate([R, c - R @ c], 1)
            return cv2.warpAffine(image, H, image.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)

class Blur(Transform):
    """Blur with given sigma (can be axis dependent)."""
    def __init__(self, sigma, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            sz = image.shape[2:]
            im1 = F.conv2d(image.view(-1,1,sz[0],sz[1]), self.filter[0], padding=(self.filter_size[0],0))
            return self.crop_to_output(F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(1,-1,sz[0],sz[1]))
        else:
            raise NotImplementedError
####

#ltr/models/layers/activation
def softmax_reg(x: torch.Tensor, dim, reg=None):
    """Softmax with optinal denominator regularization."""
    if reg is None:
        return torch.softmax(x, dim=dim)
    dim %= x.dim()
    if isinstance(reg, (float, int)):
        reg = x.new_tensor([reg])
    reg = reg.expand([1 if d==dim else x.shape[d] for d in range(x.dim())])
    x = torch.cat((x, reg), dim=dim)
    return torch.softmax(x, dim=dim)[[slice(-1) if d==dim else slice(None) for d in range(x.dim())]]
####

#pytracking/libs/tensorlist
class TensorList(list):
    """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

    def __init__(self, list_of_tensors = None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __deepcopy__(self, memodict={}):
        return TensorList(copy.deepcopy(list(self), memodict))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorList([other + e for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 - e2 for e1, e2 in zip(self, other)])
        return TensorList([e - other for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 - e1 for e1, e2 in zip(self, other)])
        return TensorList([other - e for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorList([e * other for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorList([other * e for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 / e2 for e1, e2 in zip(self, other)])
        return TensorList([e / other for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 / e1 for e1, e2 in zip(self, other)])
        return TensorList([other / e for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 @ e2 for e1, e2 in zip(self, other)])
        return TensorList([e @ other for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 @ e1 for e1, e2 in zip(self, other)])
        return TensorList([other @ e for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 % e2 for e1, e2 in zip(self, other)])
        return TensorList([e % other for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 % e1 for e1, e2 in zip(self, other)])
        return TensorList([other % e for e in self])

    def __pos__(self):
        return TensorList([+e for e in self])

    def __neg__(self):
        return TensorList([-e for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 <= e2 for e1, e2 in zip(self, other)])
        return TensorList([e <= other for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 >= e2 for e1, e2 in zip(self, other)])
        return TensorList([e >= other for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self

        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorList\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))
####

#pytracking.features.preprocessing
def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()

def sample_patch_transformed(im, pos, scale, image_sz, transforms, is_mask=False):
    """Extract transformed image samples.
    args:
        im: Image.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche
    im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz, is_mask=is_mask)

    # Apply transforms
    im_patches = torch.cat([T(im_patch, is_mask=is_mask) for T in transforms])

    return im_patches


def sample_patch_multiscale(im, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
    """Extract image patches at multiple scales.
    args:
        im: Image.
        pos: Center position for extraction.
        scales: Image scales to extract image patches from.
        image_sz: Size to resize the image samples to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """
    if isinstance(scales, (int, float)):
        scales = [scales]

    # Get image patches
    patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=mode,
                                                max_scale_change=max_scale_change) for s in scales))
    im_patches = torch.cat(list(patch_iter))
    patch_coords = torch.cat(list(coord_iter))

    return  im_patches, patch_coords

def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) // df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl//2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord
####

def load_network(network_dir=None, checkpoint=None, constructor_fun_name=None, constructor_module=None, **kwargs):
    """Loads a network checkpoint file.
    """
    #print("network_dir = " + network_dir)
    net_path = Path(network_dir)
    checkpoint = str(net_path)
    #print('checkpoint = ' + checkpoint)
    try:
        checkpoint_path = os.path.expanduser(checkpoint)
    except:
        print("couldn't load checkpoint" + checkpoint)
        raise TypeError

    # Load network
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # Construct network model
    if 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
        net_constr = checkpoint_dict['constructor']
        net = net_constr.get()
    else:
        raise RuntimeError('No constructor for the given network.')

    net.load_state_dict(checkpoint_dict['net'])

    net.constructor = checkpoint_dict['constructor']
    if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
        net.info = checkpoint_dict['net_info']

    return net, checkpoint_dict

class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter = 0

    def __init__(self, net_path, use_gpu=True, initialize=False, **kwargs):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None
        self.net_kwargs = kwargs
        if initialize:
            self.initialize()

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        # self.net = load_network(self.net_path, **self.net_kwargs)
        self.net, _ = load_network(self.net_path, **self.net_kwargs)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self):
        self.load_network()

class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""

    def __init__(self, net_path, use_gpu=True, initialize=False, image_format='rgb',
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(net_path, use_gpu, initialize, **kwargs)

        self.image_format = image_format
        self._mean = torch.Tensor(mean).view(1, -1, 1, 1)
        self._std = torch.Tensor(std).view(1, -1, 1, 1)

    def initialize(self, image_format='rgb', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().initialize()

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        if self.image_format in ['rgb', 'bgr']:
            im = im / 255

        if self.image_format in ['bgr', 'bgr255']:
            im = im[:, [2, 1, 0], :, :]
        im -= self._mean
        im /= self._std

        if self.use_gpu:
            im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        im = self.preprocess_image(im)
        return self.net.extract_backbone_features(im)

class DiMP:
    def __init__(self, params):
        self.params = params
        self.visdom = None
    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net
        '''
        print("pytracking/tracker/dimp/dimp.py")
        print("self.net = " + str(self.net))
        '''
        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)
        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        if self.frame_num == 2:
            t = (backbone_feat['layer3'].cpu()).numpy()
            with open('test.txt', 'w') as outfile:
                # We write this header for readable, the pound symbol
                # will cause numpy to ignore it
                outfile.write('# Array shape: {0}\n'.format(t.shape))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case.
                # Because we are dealing with 4D data instead of 3D data,
                # we need to add another for loop that's nested inside of the
                # previous one.
                for threeD_data_slice in t:
                    for twoD_data_slice in threeD_data_slice:
                        # The formatting string indicates that I'm writing out
                        # the values in left-justified columns 7 characters in width
                        # with 2 decimal places.
                        np.savetxt(outfile, twoD_data_slice, fmt='%-7.4f')
                        # Writing out a break to indicate different slices...
                        outfile.write('# New slice\n')
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw = self.classify_target(test_x)
        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)

        new_pos = sample_pos[scale_ind, :] + translation_vec

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)

                self.refine_target_box(backbone_feat, sample_pos[scale_ind, :], sample_scales[scale_ind], scale_ind,
                                       update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat(
            (sample_coords[scale_ind, [1, 0]], sample_coords[scale_ind, [3, 2]] - sample_coords[scale_ind, [1, 0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()
        out = {'target_bbox': output_state}
        return out

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5 * (sample_coord[:, :2] + sample_coord[:, 2:] - 1)
        sample_scales = ((sample_coord[:, 2:] - sample_coord[:, :2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2 * self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1, 1, score_filter_ksz, score_filter_ksz)
            scores = F.conv2d(scores.view(-1, 1, *scores.shape[-2:]), kernel, padding=score_filter_ksz // 2).view(
                scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1) / 2
        max_score, max_disp = max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]
        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
                output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind, :]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1 - prev_target_vec) ** 2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2 - prev_target_vec) ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change',
                                                                                            None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)

    def get_rand_shift(self, global_shift=torch.zeros(2)):
        torch.manual_seed(1)
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            return ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()
        return None

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        '''
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: (
                    (torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()
        '''

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend(
                [Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend(
                [Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(FlipHorizontal(aug_output_sz, self.get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend(
                [Blur(sigma, aug_output_sz, self.get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend(
                [Scale(scale_factor, aug_output_sz, self.get_rand_shift()) for scale_factor in
                 augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend(
                [Rotate(angle, aug_output_sz, self.get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz,
                                              self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos,
                                                         self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0], :] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

    def update_memory(self, sample_x: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos,
                                                         self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (
                        Identity, Translation, FlipHorizontal,FlipVertical,Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor(
                [self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1, 4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0], ...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            x = torch.cat([x, F.dropout2d(x[0:1, ...].expand(num, -1, -1, -1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = hann2d_clipped(self.output_sz.long(), (
                        self.output_sz * self.params.effective_search_area / self.params.search_area_scale).long(),
                                                        centered=True).to(self.params.device)
            else:
                self.output_window = hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                     'Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor',
                              self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                         'Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale=True):
        """Run the ATOM IoUNet to refine the target bounding box."""
        torch.manual_seed(1)
        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat(
                [self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min() / 3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:, 2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:] / 2) + rand_bb[:, :2]
            init_boxes = torch.cat([new_center - new_sz / 2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1, 4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)
        # print("output_boxes:")
        # print(output_boxes)
        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:, 2] / output_boxes[:, 3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (
                aspect_ratio > 1 / self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind, :]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)

    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))

    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]],
                                       device=self.params.device).view(1, 1, 4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1, 4).cpu(), outputs.detach().view(-1).cpu()

    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                self.params.device).view(1, 1, 4)

        sz_norm = output_boxes[:, :1, 2:].clone()
        output_boxes_rel = rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1, 4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale=True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1, 4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


class DBLoader(object):
    """ Debug Data Loader """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gt_file = os.path.join(self.data_dir, 'groundtruth.txt')
        self.abs_file = os.path.join(self.data_dir, 'absence.label')
        self.curr_idx = 0
        self.im_paths = glob.glob(os.path.join(self.data_dir, '*.jpg'))
        self.im_paths.sort()
        self.init_box = self.get_init_box()

    def get_init_box(self):
        # im_path = self.im_paths[0]
        # first_frame = cv2.imread(im_path)
        # init_box = cv2.selectROI(os.path.basename(self.data_dir), first_frame, False, False)
        init_box = np.array(np.loadtxt(self.gt_file, delimiter=',', dtype=np.float64))
        if len(init_box.shape) == 2:
            return init_box[0]
        return init_box

    def get_gt_box(self):
        init_box = np.array(np.loadtxt(self.gt_file, delimiter=',', dtype=np.float64))
        return init_box

    def get_abs_info(self):
        vis = np.array(np.loadtxt(self.abs_file, dtype=np.bool_))
        return vis

    def region(self):
        '''
        print("init_box:")
        print(self.init_box)
        '''
        return self.init_box

    def frame(self):
        im_path = self.im_paths[self.curr_idx] if self.curr_idx < len(self.im_paths) else None
        # print('pumping {}'.format(im_path))
        self.curr_idx += 1
        return im_path, None


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h


def get_dimp(img, init_box, model_path):
    """ set up DiMPsuper as the base tracker """
    params = TrackerParams()
    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 22 * 16
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1 / 3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False  # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3  # Top-k average to estimate final box
    params.num_init_random_boxes = 9  # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1  # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5  # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6  # Limit on the aspect ratio
    params.box_refinement_iter = 10  # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3  # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1  # Multiplicative step length decay (1 means no decay)

    params.net = NetWithBackbone(net_path=model_path,
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'
    params.visualization = True
    params.debug = False
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = DiMP(params)

    H, W, _ = img.shape
    cx, cy, w, h = get_axis_aligned_bbox(np.array(init_box))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    '''Initialize'''
    gt_bbox_np = np.array(gt_bbox_)
    gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
    init_info = {}
    init_info['init_bbox'] = gt_bbox_torch
    tracker.initialize(img, init_info)

    return tracker


def get_ar(img, init_box, ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0
    input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))
    return RF_module


def save_bb(file, data):
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, fmt="%d", delimiter=',')


def demo(base_path, ar_path, data_dir, use_ar=True):
    debug_loader = DBLoader(data_dir=data_dir)

    handle = debug_loader
    init_box = handle.region()
    gt = handle.get_gt_box()
    vis = handle.get_abs_info()
    imagefile, _ = handle.frame()
    img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    H, W, _ = img.shape

    """ Step 1: set up base tracker and Alpha-Refine """
    tracker = get_dimp(img, init_box, base_path)
    if use_ar:
        RF_module = get_ar(img, init_box, ar_path)
    file = open("tracking res/" + os.path.split(data_dir)[1] + ".txt", "w")
    res = [init_box]
    ts = []
    # OPE tracking
    while True:
        start = time.time()
        imagefile, _ = handle.frame()
        if not imagefile:
            break
        img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        """ Step 2: base tracker prediction """
        # track with base tracker
        outputs = tracker.track(img)
        pred_bbox = outputs['target_bbox']

        if use_ar:
            """ Step 3: refine tracking results with Alpha-Refine """
            pred_bbox = RF_module.refine(img, np.array(pred_bbox))

        """ Step 4: update base tracker's state with refined result """
        # x1, y1, w, h = pred_bbox.tolist()
        x1, y1, w, h = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())

        tracker.pos = new_pos.clone()
        tracker.target_sz = new_target_sz
        tracker.target_scale = new_scale

        res.append(pred_bbox)

        ts.append(time.time() - start)
    save_bb(file, res)
    file.close()
    fps = len(ts) / sum(t for t in ts)
    return fps, get_iou(np.array(res), gt, vis)

def get_iou(output, tr, vis):
    n = min(tr.shape[0], output.shape[0])
    iou = []
    for i in range(1, n):
        if vis[i] == False:
            x_l = max(tr[i][0], output[i][0])
            y_top = max(tr[i][1], output[i][1])
            x_r = min(tr[i][0] + tr[i][2], output[i][0] + output[i][2])
            y_bot = min(tr[i][1] + tr[i][3], output[i][1] + output[i][3])
            if x_r < x_l or y_bot < y_top:
                iou.append(0.0)
            else:
                inter = (x_r - x_l) * (y_bot - y_top)
                iou.append(inter / (tr[i][2] * tr[i][3] + output[i][2] * output[i][3] - inter))
    # print(iou)
    return np.mean(iou)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('model_path', type=str, default=None, help='Path to the directory with models')
    parser.add_argument('dataset_path', type=str, default=None, help='Path to the dataset')
    parser.add_argument('use_refine', type=str, default=None, help='Use Refine module')
    args = parser.parse_args()
    use_ar = args.use_refine.lower() in ("yes", "true", "t", "1")
    has_folders = 0
    '''
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    '''
    for f in os.listdir(args.dataset_path):
        if os.path.isdir(os.path.join(args.dataset_path, f)):
            has_folders = 1
            break
    # path to video sequence - any directory with series of images will be OK
    # data_dir = '../Downloads/Stark-main/data/got10k/test/kolya2'

    # path to model_file of base SEcmnet_ep0040.pthtracker - model can be download from:
    # https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv
    base_path = os.path.join(args.model_path, 'new_DiMPnet_ep0005.pth.tar')

    # path to model_file of Alpha-Refine - the model can be download from
    # https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view
    ar_path = os.path.join(args.model_path, 'SEcmnet_ep0040-c.pth.tar')
    # r'C:\Users\zadorozhnyy.v\AlphaRefine\checkpoints\ltr\dimp\super_dimp/DiMPnet_ep0005.pth.tar'

    if has_folders == 1:
        n = 0
        l = 0
        fps = []
        for f in os.listdir(args.dataset_path):
            data_path = os.path.join(args.dataset_path, f)
            if os.path.isdir(data_path):
                n += 1
                print("sequence: " + data_path)
                t, iou = demo(base_path, ar_path, data_path, use_ar)
                print("IOU = {}".format(iou))
                print('FPS: {}'.format(t))
                l += iou
                fps.append(t)
        print("Average IOU = {}".format(l / n))
        print("Average FPS = {}".format(np.mean(np.array(fps))))
    else:
        t, iou = demo(base_path, ar_path, args.dataset_path, use_ar)
        print("IOU = {}".format(iou))
        print("FPS = {}".format(t))
