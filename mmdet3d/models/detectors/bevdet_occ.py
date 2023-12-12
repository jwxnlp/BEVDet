# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D
from .aspp3d import ASPP3D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss as mmdet_build_loss
from mmseg.models.builder import build_loss as mmseg_build_loss
from mmdet3d.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
import torch.nn.functional as F

def reweight(x, beta=0.9):
    """"""
    return (1 - beta)/(1 - np.power(beta, x))

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None, # CrossEntropyLoss
                 out_dim=32,
                 beta=0.9, #
                 normalize_effective_num=1e4, #
                 aspp3d=None,
                 use_mask=False, # True
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d')) if aspp3d is None else ASPP3D(**aspp3d)
        
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.beta = beta
        self.normalize_effective_num = normalize_effective_num
        self.num_classes = num_classes
            
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.class_names = [
            "void/ignore",
            "barrier", "bicycle", "bus", "car", "construction_vehicle",
            "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
            "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
            "vegetation",
            "free"
        ]
        assert len(self.class_names) == self.num_classes, "unequal!"
        self.class_frequencies = np.array([
            1163161, 
            2309034, 188743, 2997643, 20317180, 852476,
            243808, 2457947, 497017, 2731022, 7224789,
            214411435, 5565043, 63191967, 76098082, 128860031,
            141625221,
            2307405309], dtype=np.float32)
        assert len(self.class_frequencies) == self.num_classes, "unequal!"
        self.class_weights = torch.from_numpy(
            (1-self.beta) / (1 - np.power(self.beta, 
                self.class_frequencies/self.class_frequencies.sum()*self.normalize_effective_num))).to(torch.float32)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # self.loss_occ = build_loss(loss_occ)
        # self.loss_occ = dict([(loss_name, build_loss(loss_dict)) 
        #                       for loss_name, loss_dict in loss_occ.items()])
        for loss_name, loss_dict in loss_occ.items():
            if loss_name in ["loss_ce"]:
                loss = mmdet_build_loss(loss_dict)
            elif loss_name in ["loss_dice", "loss_scal"]:
                if loss_dict.pop("use_class_weight", False):
                    loss_dict.update(dict(
                        class_weight=self.class_weights))
                loss = build_loss(loss_dict)
            else:
                raise Exception("ERROR: [ {} ]: Novalid Loss!".format(loss_name))
            self.__setattr__(loss_name, loss)
        return

    def loss_single(self,voxel_semantics,mask_camera,preds):
        """
        params:
            voxel_semantics: [B, X, Y, Z]
            mask_camera: [B, X, Y, Z]
            preds: [B, X, Y, Z, num_classes]
        """
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_
    
    def loss(self,voxel_semantics,mask_camera,preds):
        """
        params:
            voxel_semantics: [B, X, Y, Z]
            mask_camera: [B, X, Y, Z]
            preds: [B, X, Y, Z, num_classes]
        """
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        B, X, Y, Z = voxel_semantics.shape
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32) # [B, X, Y, Z]
            # voxel_semantics=voxel_semantics.reshape(-1)
            # preds=preds.reshape(-1,self.num_classes)
            # mask_camera = mask_camera.reshape(-1)
            # num_total_samples=mask_camera.sum()
            class_weights = self.class_weights.to(preds.device)
            # print("--- class_weights: ")
            
            # cross entropy loss
            mask = mask_camera.reshape(-1) * class_weights[voxel_semantics].reshape(-1)
            # num_total_samples=mask.sum()
            loss_ce = self.__getattr__('loss_ce')(
                preds.reshape(-1,self.num_classes),
                voxel_semantics.reshape(-1),
                mask, avg_factor=mask.sum())
            loss_['loss_ce'] = loss_ce
            # dice loss
            # dice loss from mmdet
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # loss_dice=self.__getattr__('loss_dice')(
            #     F.softmax(preds, dim=-1) * mask_camera[..., None].expand(-1, -1, -1, -1, self.num_classes),
            #     F.one_hot(voxel_semantics, num_classes=self.num_classes)  * mask_camera[..., None].expand(-1, -1, -1, -1, self.num_classes))
            
            # dice loss from mmseg
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # [B*N_view, H_L4, W_L4, 1, S_L4, S_L4]
            # gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
            
            loss_dice = self.__getattr__('loss_dice')(
                preds, voxel_semantics, mask_camera)
            loss_['loss_dice'] = loss_dice
            
            # Scene Class Affinity Loss
            loss_scal = self.__getattr__('loss_scal')(
                preds, voxel_semantics, mask_camera)
            loss_['loss_scal'] = loss_scal
            
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_ce = self.loss_occ['loss_ce'](preds, voxel_semantics,)
            loss_['loss_ce'] = loss_ce
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None, #
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None, #
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # kwargs keys: gt_depth, voxel_semantics, mask_lidar, mask_camera
        # img_feats[0]: [B, C, Z, Y, X]
        # depth: [B*N_view, D, H_L4, W_L4]
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth'] # [B, N_view, H_in, W_in]
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth # weight = 0.05
        # [B, X, Y, Z, C]
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            # [B, X, Y, Z, num_classes]
            occ_pred = self.predicter(occ_pred)
        
        # [B, X, Y, Z]
        voxel_semantics = kwargs['voxel_semantics'] # torch.uint8
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
