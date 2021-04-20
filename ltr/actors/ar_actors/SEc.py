from ltr.actors import BaseActor
import torch


class SEc_Actor(BaseActor):
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
        corner_pred = pred_dict['corner']  # (batch,4)

        '''get groundtruth'''
        bbox_gt = data['test_anno'].squeeze(0)  # 测试帧的真值框在裁剪出的搜索区域上的坐标(x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)格式

        '''get loss function'''
        corner_loss_F = self.objective['corner']

        '''Compute loss for corner'''
        loss_corner = corner_loss_F(corner_pred, bbox_gt_xyxy)
        loss = loss_corner

        stats = {
            'Loss/total': loss.item(),
            'loss_corner': loss_corner.item(),
        }

        return loss, stats
