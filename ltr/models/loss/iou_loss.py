# GIoU and Linear IoU are added by following
# https://github.com/yqyao/FCOS_PLUS/blob/master/maskrcnn_benchmark/layers/iou_loss.py.
import torch
from torch import nn

'''上面列出的链接里给的是当边界框以(l,t,r,b)形式给出时的iou loss,而我们要做的是(x1,y1,x2,y2)格式边界框的iou loss'''
class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type
    '''new version for (x1,y1,x2,y2) rather than (l,t,r,b)'''
    def forward(self, pred, target, weight=None):
        '''form: (x1,y1,x2,y2)'''
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 2]
        pred_y2 = pred[:, 3]

        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]
        '''Compute seperate areas'''
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        '''Compute intersection area and iou'''
        w_inter = torch.clamp(torch.min(pred_x2,target_x2) - torch.max(pred_x1,target_x1) , 0)
        h_inter = torch.clamp(torch.min(pred_y2,target_y2) - torch.max(pred_y1,target_y1) , 0)
        area_intersect = w_inter * h_inter
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        '''Compute C and Giou'''
        W_c = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        H_c = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        ac_uion = W_c * H_c + 1e-7
        extra_term = (ac_uion - area_union) / ac_uion
        gious = ious - extra_term
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return (losses.sum(),ious.sum(),extra_term.sum())

'''上面的iou loss是求sum,但是这就造成其结果跟batchsize有关,感觉不太合适. 这个版本是求mean,这样就不会跟batchsize有关了'''
class IOULoss_mean(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss_mean, self).__init__()
        self.loss_type = loss_type
    '''new version for (x1,y1,x2,y2) rather than (l,t,r,b)'''
    def forward(self, pred, target, weight=None):
        '''form: (x1,y1,x2,y2)'''
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 2]
        pred_y2 = pred[:, 3]

        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]
        '''Compute seperate areas'''
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        '''Compute intersection area and iou'''
        w_inter = torch.clamp(torch.min(pred_x2,target_x2) - torch.max(pred_x1,target_x1) , 0)
        h_inter = torch.clamp(torch.min(pred_y2,target_y2) - torch.max(pred_y1,target_y1) , 0)
        area_intersect = w_inter * h_inter
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        '''Compute C and Giou'''
        W_c = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        H_c = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        ac_uion = W_c * H_c + 1e-7
        extra_term = (ac_uion - area_union) / ac_uion
        gious = ious - extra_term
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            '''use mean rather than sum'''
            return (losses.mean(),ious.mean(),extra_term.mean())