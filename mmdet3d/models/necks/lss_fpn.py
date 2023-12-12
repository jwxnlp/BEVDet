# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmdet3d.models.backbones.resnet import ConvModule
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x

@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels, #
                 out_channels, #
                 with_cp=False):
        super().__init__()
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x

@NECKS.register_module()
class CustomLSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels, #
                 out_channels, #
                 with_cp=False):
        super().__init__()
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)
        self.up3 =  nn.Upsample(
            scale_factor=8, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32, x_64 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x_64 = self.up3(x_64)
        x = torch.cat([x_8, x_16, x_32, x_64], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x


@NECKS.register_module()
class CustomLSSDBFPN3D(nn.Module):
    def __init__(self,
                 in_channels, #
                 out_channels, #
                 conv_cfg=dict(type='Conv3d'), # dict(type='Conv3d')
                 norm_cfg=dict(type='BN3d', ), # dict(type='BN3d')
                 act_cfg=dict(type='ReLU',inplace=True), # dict(type='ReLU',inplace=True)
                 with_cp=False):
        super().__init__()
        """"""
        # top down pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            if i != len(in_channels) - 1:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            if i != len(in_channels) - 1:
                self.fpn_convs.append(fpn_conv)

        self.conv = ConvModule(
            out_channels*4,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.with_cp = with_cp

    def forward(self, feats):
        """"""
        # build laterals
        laterals = [
            lateral_conv(feats[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1]+F.interpolate(
                laterals[i], size=prev_shape, mode='trilinear', align_corners=True))
        
        outs = [laterals[0]] + [
            F.interpolate(
                lateral, size=laterals[0].shape[2:], mode='trilinear', align_corners=True)
            for lateral in laterals[1:]]
        x = torch.cat(outs, dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x

@NECKS.register_module()
class CustomLSSUNetDecoder3D(nn.Module):
    def __init__(self,
                 in_channels, #
                 out_channels, #
                 conv_cfg=dict(type='Conv3d'), # dict(type='Conv3d')
                 norm_cfg=dict(type='BN3d', ), # dict(type='BN3d')
                 act_cfg=dict(type='ReLU',inplace=True), # dict(type='ReLU',inplace=True)
                 with_cp=False):
        super().__init__()
        """"""
        # top down pathway
        self.lateral_conv1s = nn.ModuleList()
        self.lateral_conv2s = nn.ModuleList()
        for i in range(len(in_channels)-1):
            l_conv1 = ConvModule(
                in_channels[i]+in_channels[i+1],
                in_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            l_conv2 = ConvModule(
                in_channels[i],
                in_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_conv1s.append(l_conv1)
            self.lateral_conv2s.append(l_conv2)

        self.conv = ConvModule(
            in_channels[0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.with_cp = with_cp

    def forward(self, feats):
        """"""
        used_backbone_levels = len(feats)
        # build top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = feats[i - 1].shape[2:]
            lateral = torch.cat([feats[i - 1], F.interpolate(
                feats[i], size=prev_shape, mode='trilinear', align_corners=True)], dim=1)
            feats[i - 1] = self.lateral_conv2s[i-1](self.lateral_conv1s[i-1](lateral))
        
        x = feats[0]
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x


@NECKS.register_module()
class CustomLSSPAN3D(nn.Module):
    def __init__(self,
                 in_channels, # [N_L0, N_L1, N_L2, N_L3]
                 out_channels, #
                 conv_cfg=dict(type='Conv3d'), # dict(type='Conv3d')
                 norm_cfg=dict(type='BN3d', ), # dict(type='BN3d')
                 act_cfg=dict(type='ReLU',inplace=True), # dict(type='ReLU',inplace=True)
                 with_cp=False):
        super().__init__()
        """"""
        # top down pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(1, len(in_channels)):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        
        # DBNeck
        self.db_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            db_conv = ConvModule(
                out_channels,
                out_channels // len(in_channels),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=norm_cfg)
            self.db_convs.append(db_conv)
            
        self.with_cp = with_cp
        return
    
    def ck(self, f, x, with_cp=False):
        """"""
        if with_cp:
            return checkpoint(f, x)
        else:
            return f(x)

    def forward(self, feats):
        """"""
        # build laterals
        laterals = [
            lateral_conv(feats[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='trilinear', align_corners=True)
        
        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])
        
        # DBNeck
        uni_outs = []
        uni_shape = outs[0].shape[2:]
        for i in range(len(outs)):
            uni_out = self.db_convs[i](outs[i])
            if i != 0:
                uni_out = F.interpolate(uni_out, size=uni_shape, 
                                        mode='trilinear', align_corners=True)
            uni_outs.append(uni_out)
            
        return torch.cat(uni_outs, dim=1)

@NECKS.register_module()
class CustomLSSPAN3DV2(nn.Module):
    def __init__(self,
                 in_channels, # [N_L0, N_L1, N_L2, N_L3]
                 out_channels, #
                 conv_cfg=dict(type='Conv3d'), # dict(type='Conv3d')
                 norm_cfg=dict(type='BN3d', ), # dict(type='BN3d')
                 act_cfg=dict(type='ReLU',inplace=True), # dict(type='ReLU',inplace=True)
                 with_cp=False):
        super().__init__()
        """"""
        # top down pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(1, len(in_channels)):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        
        self.conv = ConvModule(
            out_channels*4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.with_cp = with_cp
        return
    
    def ck(self, f, x, with_cp=False):
        """"""
        if with_cp:
            return checkpoint(f, x)
        else:
            return f(x)

    def forward(self, feats):
        """"""
        # build laterals
        laterals = [
            lateral_conv(feats[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='trilinear', align_corners=True)
        
        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

        for i in range(1, used_backbone_levels):
            inter_outs[i] = self.pafpn_convs[i - 1](inter_outs[i])
        
        outs = []
        outs.append(inter_outs[0])
        uni_shape = inter_outs[0].shape[2:]
        outs.extend([
            F.interpolate(inter_outs[i], size=uni_shape, 
                mode='trilinear', align_corners=True)
            for i in range(1, used_backbone_levels)
        ])
        
        
        x = torch.cat(outs, dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x