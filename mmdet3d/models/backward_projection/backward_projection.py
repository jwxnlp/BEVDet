# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.base_module import BaseModule
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmdet.models.utils import build_transformer



@HEADS.register_module()
class BackwardProjection(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 transformer=None, # BEVFormer
                 positional_encoding=None, # CustormLearnedPositionalEncoding
                 pc_range=None, # [-40, -40, -1.0, 40, 40, 5.4]
                 in_channels=64, # numC_Trans
                 out_channels=64, # numC_Trans
                 use_zero_embedding=False,
                 bev_h=30, # 100
                 bev_w=30, # 100
                 **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.use_zero_embedding = use_zero_embedding
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
       
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims # numC_Trans

        self._init_layers()

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, 
                mlvl_feats, # [[B, N_view, con_C, H_L4, W_L4], ], one level
                img_metas, #
                lss_bev=None, # [B, con_C, Y, X]
                gt_bboxes_3d=None, # None
                cam_params=None, #
                pred_img_depth=None, # [B, N_view, D, H_L4, W_L4]
                bev_mask=None): # None
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype) # [bev_h * bev_w, E]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # [bev_h * bev_w, B, E]
        
        if lss_bev is not None:
            lss_bev = lss_bev.flatten(2).permute(2, 0, 1) # [Y*X, B, con_C]
            # bev_queries = bev_queries + lss_bev # why not assign value
        
        if bev_mask is not None:
            bev_mask = bev_mask.reshape(bs, -1)
        # [B, 2*N_feats, bev_h, bev_w]
        bev_pos = self.positional_encoding(bs, self.bev_h, self.bev_w, bev_queries.device).to(dtype)
        # [B, bev_h * bev_w, E]
        bev =  self.transformer(
                mlvl_feats, # [[B, N_view, con_C, H_L4, W_L4], ], one level
                bev_queries, # [bev_h * bev_w, B, E]
                lss_bev, # [Y * X, B, E]
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos, # [B, 2*N_feats, bev_h, bev_w]
                img_metas=img_metas, #
                cam_params=cam_params, #
                gt_bboxes_3d=gt_bboxes_3d, # None
                pred_img_depth=pred_img_depth, # [B, N_view, D, H_L4, W_L4]
                prev_bev=None, # temporal self attention
                bev_mask=bev_mask, # None
            )
        # [B, E, bev_h, bev_w]
        bev = bev.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w).contiguous()


        return bev

