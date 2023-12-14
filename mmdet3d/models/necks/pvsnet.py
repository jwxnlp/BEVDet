# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule, auto_fp16, Sequential
from mmdet.models.backbones.resnet import BasicBlock

from ..builder import NECKS


@NECKS.register_module()
class PVSNet(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 channels, # 256 channels of L4 features through img_neck
                 block=BasicBlock,
                 num_blocks=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super(PVSNet, self).__init__(init_cfg)
        
        self.channels = channels
        self.block = block
        self.num_blocks = num_blocks
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        
        layers = []
        for i in range(1, num_blocks):
            layers.append(
                self.block(
                    inplanes=self.channels,
                    planes=self.channels,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.net = Sequential(*layers)
        
        return

    @auto_fp16()
    def forward(self, x):
        """Forward function.
        params:
            x: [B*N_view, C, H, W]
        """
        x = self.net(x)
        
        return x
