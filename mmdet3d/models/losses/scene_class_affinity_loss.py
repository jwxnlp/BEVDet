# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weighted_loss


def geo_scal_loss(pred, target, mask, semantic=True, smooth=1, class_weight=None):
    """
    args:
        pred: [B, X, Y, Z, num_classes]
        target: [B, X, Y, Z]
        mask: [B, X, Y, Z]
    """
    # Get Softmax Probabilities
    if semantic:
        pred = F.softmax(pred, dim=-1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[..., -1]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    nonempty_target = (target != 17).float() # [B, X, Y, Z], 17 corresponds to free
    # nonempty_target = nonempty_target[mask].float() # [N,]
    # prection does not need to include 255, but gt needs
    # nonempty_probs = nonempty_probs[mask]
    # empty_probs = empty_probs[mask]
    
    nonempty_target = nonempty_target.reshape(nonempty_target.shape[0], -1)
    nonempty_probs = nonempty_probs.reshape(nonempty_probs.shape[0], -1)
    empty_probs = empty_probs.reshape(empty_probs.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)

    intersection = torch.sum(nonempty_target * nonempty_probs * mask, dim=1)
    precision = intersection / torch.sum(nonempty_probs * mask, dim=1)
    recall = (intersection+smooth) / (torch.sum(nonempty_target, dim=1)+smooth)
    spec = (torch.sum((1 - nonempty_target) * (empty_probs) * mask, dim=1)+smooth) / (torch.sum(
        (1 - nonempty_target)*mask, dim=1)+smooth)
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))).mean()


def sem_scal_loss(pred, target, mask, smooth=1, class_weight=None, use_free=True):
    """
    args:
        pred: [B, X, Y, Z, num_classes]
        target: [B, X, Y, Z]
        mask: [B, X, Y, Z]
    """
    # Get softmax probabilities
    pred = F.softmax(pred, dim=-1)
    loss = 0
    count = 0
    num_classes = pred.shape[-1]
    one_hot_target = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes)
    if not use_free:
        num_classes -= 1
    for i in range(0, num_classes):

        # Get probability of class i
        p = pred[..., i]
        p = p.reshape(p.shape[0], -1)
        t = one_hot_target[..., i]
        t = t.reshape(t.shape[0], -1)
        m = mask.reshape(mask.shape[0], -1)

        
        nominator = torch.sum(p * t * m, dim=1)
        loss_class = 0
        
        precision = nominator / (torch.sum(p * m, dim=1))
        loss_precision = F.binary_cross_entropy(
            precision, torch.ones_like(precision)
        )
        loss_class += loss_precision
    
        recall = (nominator+smooth) / (torch.sum(t * m, dim=1)+smooth)
        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
        loss_class += loss_recall
    
        specificity = (torch.sum((1 - p) * (1 - t) * m, dim=1)+smooth) / (
            torch.sum((1 - t) * m, dim=1)+smooth)
        loss_specificity = F.binary_cross_entropy(
            specificity, torch.ones_like(specificity)
        )
        loss_class += loss_specificity
        if class_weight is not None:
            loss += loss_class.mean() * class_weight[i]
        else:
            loss += loss_class.mean()
    if class_weight is not None:
        return loss / class_weight[:num_classes].sum()
    else:
        return loss / num_classes


@LOSSES.register_module()
class SceneClassAffinityLoss(nn.Module):
    """SceneClassAffinityLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_scal',
                 use_free=True,
                 **kwargs):
        super(SceneClassAffinityLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.use_free = use_free
        return

    def forward(self,
                pred,
                target,
                valid_mask,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss = self.loss_weight * (sem_scal_loss(
            pred,target,mask=valid_mask.long(),smooth=self.smooth, class_weight=class_weight, use_free=self.use_free) + geo_scal_loss(
            pred,target,mask=valid_mask.long(),smooth=self.smooth, class_weight=class_weight))
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
