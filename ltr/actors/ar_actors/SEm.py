from ltr.actors import BaseActor
import torch


class SEm_Actor(BaseActor):
    """ Actor for training the our scale estimation module"""

    def delta2xyxy(self, delta):
        '''delta的size是(batch,4)'''
        bbox_cxcywh = delta.clone()
        '''以位于中心的大小为(128,128)的框为基准'''
        bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # 中心偏移
        bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # 宽高修正
        '''调整为(x1,y1,x2,y2)格式'''
        bbox_xyxy = bbox_cxcywh.clone()
        bbox_xyxy[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
        bbox_xyxy[:, 2:] = bbox_cxcywh[:, :2] + 0.5 * bbox_cxcywh[:, 2:]
        return bbox_xyxy

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain bbox prediction for each test image'
        '''get prediction'''
        pred_dict = self.net(data['train_images'], data['test_images'], data['train_anno'])
        mask_pred = pred_dict['mask']  # (batch,1,256,256)

        '''get groundtruth'''
        bbox_gt = data['test_anno'].squeeze(0)  # 测试帧的真值框在裁剪出的搜索区域上的坐标(x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)格式

        '''get loss function'''
        mask_loss_F = self.objective['mask']

        '''Compute total loss'''
        mask_flag = (data['mask'] == 1)  # data[mask]是一个tensor,有batch个元素,其中有的是0有的是1. mask_flag是一个bool型tensor
        num_mask_sample = mask_flag.sum()
        if num_mask_sample > 0:
            mask_gt = data['test_masks'].squeeze(0)  # 测试帧的mask真值框 (batch,1,H,W)
            '''Compute loss for mask'''
            loss_mask = mask_loss_F(mask_pred[mask_flag], mask_gt[mask_flag])  # 只计算那些mask_flag等于1的样本的loss_mask
            loss = 1000 * loss_mask
        else:
            loss_mask = torch.zeros((1,))
            loss = loss_mask

        stats = {
            'Loss/total': loss.item(),
            'loss_mask': loss_mask.item(),
        }

        return loss, stats
