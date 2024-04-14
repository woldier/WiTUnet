# -*- coding: utf-8 -*-
"""
@Time ： 3/22/24 5:07 AM
@Auth ： woldier wong
@File ：WiTUnet.py
@IDE ：PyCharm
@DESCRIPTION：WiTUnet
"""
import torch, torch.nn as nn
from model.our.window_transformer import WindowTransformerBlock
import torch.utils.checkpoint as checkpoint
import math
from model import AbstractDenoiser
from typing import Optional, List, Tuple, Any
from timm.models.layers import trunc_normal_
from model.our.attention import SepConv2d
from einops import rearrange


class WiTUBlock(nn.Module):
    """基本的WiTUnet Block

    Examples:
        >>> net = WiTUBlock(
        >>> dim=64,
        >>> input_resolution=(256, 256),
        >>> depth=4,
        >>> num_heads=8,
        >>> win_size=8,
        >>> token_projection='conv',
        >>> token_mlp='LiPe'
        >>>
        >>> )
        >>> # 一个特征图 [B, H, W, C]
        >>> feature_map = torch.randn((1, 256, 256, 64))
        >>> shape = feature_map.shape
        >>> # 转换[B, H*W, C]
        >>> feature_map = feature_map.view(shape[0], -1, shape[-1])
        >>> out = net(feature_map)
    """

    def __init__(
            self,
            dim: int,
            output_dim,
            input_resolution,
            depth,
            num_heads,
            win_size,
            mlp_ratio=4.,
            qkv_bias: bool = True,
            qk_scale=None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: list = 0.,
            norm_layer=nn.LayerNorm,
            use_checkpoint: bool = False,
            token_projection: str = 'linear',
            token_mlp: str = 'ffn',
            shift_flag: bool = True,
            modulator: bool = False,
            cross_modulator: bool = False
    ):
        """

        :param dim: 图片的通道数
        :param output_dim: 输出的通道数
        :param input_resolution: 输入图片的h,w
        :param depth: WindowTransformerBlock的个数
        :param num_heads: 多头的数目
        :param win_size: window的大小
        :param mlp_ratio: mlp中隐藏层是输入维度的多少倍
        :param qkv_bias: qkv的bias
        :param qk_scale: qkv的scale
        :param drop: 投影层的drop
        :param attn_drop: atten中的drop
        :param drop_path: atten后与mpl连接时的drop
        :param norm_layer: att, mlp 之前使用的nrom
        :param use_checkpoint:
        :param token_projection:  投影的类型, 可选 "linear"和"conv"
        :param token_mlp:  FFN 的类型,   可选 "mlp"和"LiPe"
        :param shift_flag: 是否要进行shift
        :param modulator:  是否需要调制器
        :param cross_modulator:  是否需要cross 调制器
        """

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # 如果说shift_flag为False,if条件断路, 那么所有的shift值都是0
        # 如果说shift_flag为True,说明要进行shift, 此时偶数层则进行shift 而基数层不会.
        shift_sizes = [0 if ((not shift_flag) or (i % 2 == 0)) else win_size // 2
                       for i in range(depth)]
        # 生成WiTUBlock
        self.blocks = nn.ModuleList([
            WindowTransformerBlock(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, win_size=win_size,
                                   shift_size=shift_sizes[i],  # shift size
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                   modulator=modulator, cross_modulator=cross_modulator)
            for i in range(depth)])
        self.out = nn.Identity() if dim == output_dim else NestedBlock(in_channels=dim, out_channels=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.out(x)
        return x


class NestedBlock(nn.Module):
    """
    中间连接层
    Example:
            >>> net = NestedBlock(in_channels=32, out_channels=16)
            >>>  # 一个特征图 [B, H, W, C]
            >>> feature_map = torch.randn((1, 256, 256, 32))
            >>> shape = feature_map.shape
            >>>  # 转换 [B, H*W, C]
            >>> feature_map = feature_map.view(shape[0], -1, shape[-1])
            >>> print(net(feature_map).shape)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(NestedBlock, self).__init__()
        self.conv = SepConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)  # 转换为图片
        x = self.conv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        return x


class DownSample(nn.Module):
    """
    下采样层
    """

    def __init__(
            self,
            in_channel: int,
            out_channel: int
    ):
        """

        :param in_channel: 输入的通道数
        :param out_channel:  输出的通道数
        """
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class UpSample(nn.Module):
    """
    上采样层
    """

    def __init__(
            self,
            in_channel: int,
            out_channel: int
    ):
        """

        :param in_channel: 输入的通道数
        :param out_channel:  输出的通道数
        """
        super(UpSample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WiTUnet(AbstractDenoiser):
    def __init__(self, img_size: int = 256, in_chans: int = 3, dd_in: int = 3,
                 embed_dim: int = 32, depths: Optional[list] = None, num_heads: Optional[list] = None,
                 win_size: int = 8, mlp_ratio: float = 4., qkv_bias: bool = True, qk_scale=None,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_norm: bool = True,
                 use_checkpoint: bool = False, token_projection: str = 'linear', token_mlp: str = 'LiPe',
                 shift_flag: bool = True, modulator: bool = False, cross_modulator: bool = False,
                 need_out_relu: bool = False, nested: bool = True, deep_super: bool = False
                 ):
        """
        :param img_size: 图片的尺寸, 长宽一致
        :param in_chans:  最终输出的通道数
        :param dd_in: 输入图片的通道数
        :param embed_dim: 特征层 的基本维度c
        :param depths: 深度数组, 数组中的值保存的是一个ublock中window_trans的个数.
            depths // 2 即为encoder 或者decoder 的个数. 数组的个数一定为奇数, 最中间的元素指的是Bottleneck中的window_trans个数
        :param num_heads: 多头的个数, 每个ublock中多头的数量.
        :param win_size: 窗口的大小
        :param mlp_ratio: mlp中隐藏层是输入维度的多少倍
        :param qkv_bias: qkv的bias
        :param qk_scale: qkv的scale
        :param drop_rate: 投影层的drop
        :param attn_drop_rate: atten中的drop
        :param drop_path_rate: atten后与mpl连接时的drop
        :param norm_layer: att, mlp 之前使用的nrom
        :param patch_norm:
        :param use_checkpoint:
        :param token_projection: 投影的类型, 可选 "linear"和"conv"
        :param token_mlp:  FFN 的类型,   可选 "mlp"和"LiPe"
        :param shift_flag: 是否要进行shift
        :param modulator: 是否需要调制器
        :param cross_modulator: 是否需要cross 调制器
        :param need_out_relu:  最后的输出是否需要relu
        :param nested:  是否需要Unet++ pathway 操作
        :param deep_super:  是否使用深监督, 默认为False
        """
        super(WiTUnet, self).__init__(loss_function=nn.MSELoss())
        if num_heads is None:
            num_heads = [1, 2, 4, 8, 16, 16, 8, 4, 2]
        if depths is None:
            depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        assert len(depths) % 2 == 1, "the depth list len must be odd!"
        self.is_nested = nested
        self.num_heads = num_heads
        self.depths = depths
        coder_num = len(depths) // 2  # encoder or decoder num
        self.coder_num = coder_num
        self.num_enc_layers = coder_num  # 得到encoder的层数
        self.num_dec_layers = coder_num  # 得到decoder的层数
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.use_checkpoint = use_checkpoint
        self.token_projection = token_projection
        self.token_mlp = token_mlp
        self.shift_flag = shift_flag
        self.modulator = modulator
        self.cross_modulator = cross_modulator

        self.token_projection = token_projection  # token 投影的方式
        self.mlp = token_mlp  # 采用的mlp/ffn 的类型
        self.win_size = win_size  # 出卬叩的大小
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        self.need_relu = need_out_relu
        self.is_deep_super = deep_super
        # stochastic depth
        # 统计encoder 中有多少个 window_attn block , 每个block中的drop path比率一次增加
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[coder_num]  # Bottleneck 的drop path 比率
        dec_dpr = enc_dpr[::-1]  # decoder中的drop path 比率 逐渐减小

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        # deep supervise out proj
        if self.is_deep_super:
            deep_super_out_proj = nn.ModuleList([])
            for _ in range(self.coder_num - 1):
                deep_super_out_proj.append(
                    OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
                )
            self.deep_super_out_proj = deep_super_out_proj
        # 构建encoder
        encoder_in_channels = [2 ** i for i in range(self.coder_num)]
        encoder_out_channels = [2 ** (i + 1) for i in range(self.coder_num)]
        self._init_encoder(enc_dpr, encoder_in_channels, encoder_out_channels)

        # Bottleneck
        rate = encoder_out_channels[-1]
        self._init_bottleneck(conv_dpr, rate)

        # 如果有nested 操作, 那么UpBlock需要接收的额外的通道信息
        additional_channels = self._init_pathway()

        # 构建decoder
        decoder_in_channels = [encoder_out_channels[-1]] + encoder_out_channels[-1:0:-1]
        decoder_out_channels = encoder_in_channels[::-1]
        self._init_decoder(dec_dpr, decoder_in_channels, decoder_out_channels, additional_channels)

        self.apply(self._init_weights)

    def _init_decoder(self, dec_dpr, decoder_in_channels, decoder_out_channels, additional_channels):
        decoders = nn.ModuleList([])
        offset = self.coder_num + 1
        for idx, (in_c, out_c) in enumerate(zip(decoder_in_channels, decoder_out_channels)):
            upsample = UpSample(self.embed_dim * in_c, self.embed_dim * out_c)
            decoders.append(upsample)
            decoderlayer = WiTUBlock(
                dim=self.embed_dim * out_c * (2 + additional_channels[idx]),  # 由于拼接, dim 翻倍了
                output_dim=self.embed_dim * out_c * 2,
                input_resolution=(self.reso // out_c,
                                  self.reso // out_c),
                depth=self.depths[offset + idx],
                num_heads=self.num_heads[offset + idx],
                win_size=self.win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,

                drop_path=dec_dpr[sum(self.depths[offset:offset + idx]):sum(self.depths[offset:offset + idx + 1])],
                norm_layer=self.norm_layer,
                use_checkpoint=self.use_checkpoint,
                token_projection=self.token_projection, token_mlp=self.token_mlp,
                shift_flag=self.shift_flag,
                modulator=self.modulator, cross_modulator=self.cross_modulator
            )
            decoders.append(decoderlayer)
        self.decoders = decoders

    def _init_pathway(self):
        additional_channels = [0 for _ in range(self.coder_num)]
        if self.is_nested:
            # 构建nested
            nested_blocks = nn.ModuleList([])
            for i in range(self.coder_num - 1):
                for c in range(i + 1):
                    exp = i - c
                    dims = self.embed_dim * (2 ** exp)
                    in_dim, out_dim = dims * (c + 2), dims
                    nested_blocks.append(
                        NestedBlock(in_channels=in_dim, out_channels=out_dim)
                    )
            self.nested_blocks = nested_blocks
            additional_channels = [i for i in range(self.coder_num)]
            # 构建额外的UpSample
            encoder_up = nn.ModuleList([])
            for i in range(self.coder_num - 1):
                for j in range(i + 1):
                    exp = i - j
                    dims = self.embed_dim * (2 ** exp)
                    encoder_up.append(
                        UpSample(dims * 2, dims)
                    )
            self.encoder_up = encoder_up
        return additional_channels

    def _init_bottleneck(self, conv_dpr, rate):
        self.conv = WiTUBlock(
            dim=self.embed_dim * rate,
            output_dim=self.embed_dim * rate,
            input_resolution=(self.reso // rate,
                              self.reso // rate),
            depth=self.depths[self.coder_num],  # Bottleneck 的depth
            num_heads=self.num_heads[self.coder_num],  # Bottleneck 的num_heads
            win_size=self.win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
            drop=self.drop_rate, attn_drop=self.attn_drop_rate,
            drop_path=conv_dpr,
            norm_layer=self.norm_layer,
            use_checkpoint=self.use_checkpoint,
            token_projection=self.token_projection, token_mlp=self.token_mlp, shift_flag=self.shift_flag)

    def _init_encoder(self, enc_dpr, encoder_in_channels, encoder_out_channels):
        encoders = nn.ModuleList([])
        for idx, (in_c, out_C) in enumerate(zip(encoder_in_channels, encoder_out_channels)):
            encoderlayer = WiTUBlock(
                dim=self.embed_dim * in_c,
                output_dim=self.embed_dim * in_c,
                input_resolution=(self.reso // in_c,  # feature map 的H W
                                  self.reso // in_c
                                  ),
                depth=self.depths[idx],
                num_heads=self.num_heads[idx],
                win_size=self.win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                # 第 idx 个 WiTUBlock 中 window attn 对于path drop 的参数
                drop_path=enc_dpr[sum(self.depths[:idx]):sum(self.depths[:idx + 1])],
                norm_layer=self.norm_layer,
                use_checkpoint=self.use_checkpoint,
                token_projection=self.token_projection, token_mlp=self.token_mlp,
                shift_flag=self.shift_flag
            )
            encoders.append(encoderlayer)
            dowsample = DownSample(self.embed_dim * in_c, self.embed_dim * out_C)
            encoders.append(dowsample)
        self.encoders = encoders

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_loss(self, pre: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_function(pre, target)
        return loss

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> (torch.Tensor, Optional[torch.Tensor]):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv_res = []
        pool_res = []
        temp = y
        for net in self.encoders:
            temp = net(temp)
            if len(conv_res) <= len(pool_res):  # 如果conv_res中元素个数与pool_res相等那么向conv_res中添加元素
                conv_res.append(temp)
            else:
                pool_res.append(temp)
        # deep supervise res
        deep_super_res = []
        if self.is_nested:
            assert self.encoder_up is not None, "self.encoder_up is None!"
            assert self.nested_blocks is not None, "self.nested_blocks is None!"
            conv_res_nested = [[res] for res in conv_res]
            for i in range(self.coder_num - 1):
                nest_temp = conv_res[1 + i]  # 得到encoder 当前层的下一层, 以进行上采样
                offset = sum([n + 1 for n in range(i)])
                for j in range(i + 1):
                    enc_up = self.encoder_up[offset + j]  # 得到上采样block
                    nest_temp = enc_up(nest_temp)  # 上采样
                    # 拼接来自下一级和前面的跳跃连接
                    nest_temp = torch.cat(conv_res_nested[i - j][:] + [nest_temp], dim=-1)
                    nest_temp = self.nested_blocks[offset + j](nest_temp)
                    conv_res_nested[i - j].append(nest_temp)
            conv_res = [torch.cat(item, dim=-1) for item in conv_res_nested]  # 形成新的conv_res
            deep_super_res = conv_res_nested[0][1:]
        # Bottleneck
        temp = self.conv(temp)

        for i, conv in enumerate(conv_res[::-1]):
            up = self.decoders[i * 2]
            decoderlayer = self.decoders[i * 2 + 1]
            temp = up(temp)
            temp = torch.cat([temp, conv], -1)
            temp = decoderlayer(temp)

        # Output Projection
        y = self.output_proj(temp)
        if self.need_relu:
            pre = torch.relu(x + y) if self.dd_in == 1 else torch.relu(y)
        else:
            pre = x + y if self.dd_in == y.shape[1] else y
        # 计算 损失
        # 如果是训练模式 self.training 为True, 那么 if 短路, 一定会进入if内部, 此时必须检查target是否是None
        # 还有种情况, 是处理eval模式, 但是我们仍然想看一下此时的损失, 那么就需要传入target, 此时会进行计算loss
        loss = None
        if self.training or target is not None:
            assert target is not None, "model in train mod, the target can't be None!"
            loss = self.loss_function(pre, target)
            if self.is_nested and self.is_deep_super:  # 有pathway 并且是 深监督
                assert len(self.deep_super_out_proj) == len(deep_super_res), \
                    f"the length of deep_super_res:{len(deep_super_res)}, not equals deep_super_out_proj:{len(self.deep_super_out_proj)}"
                # 前面的punish 少一些, 后面的 punish 多一些
                inc_loss_rate = torch.linspace(1e-2, 1, len(self.deep_super_out_proj))
                for i, (proj, rate) in enumerate(zip(self.deep_super_out_proj, inc_loss_rate)):
                    pre_super = proj(deep_super_res[i])
                    loss += rate * self.loss_function(pre_super, target)
        return pre, loss


if __name__ == '__main__':
    # net = WiTUBlock(
    #     dim=64,
    #     input_resolution=(256, 256),
    #     depth=4,
    #     num_heads=8,
    #     win_size=8,
    #     token_projection='conv',
    #     token_mlp='LiPe'
    #
    # )
    # # 一个特征图 [B, H, W, C]
    # feature_map = torch.randn((1, 256, 256, 64))
    # shape = feature_map.shape
    # # 转换[B, H*W, C]
    # feature_map = feature_map.view(shape[0], -1, shape[-1])
    # out = net(feature_map)

    # net = WiTUnet(img_size=512, in_chans=1, dd_in=1).cuda()
    # input = torch.randn((1, 1, 512, 512)).cuda()
    net = WiTUnet(img_size=512, in_chans=1, dd_in=1, deep_super=True)
    input = torch.randn((1, 1, 512, 512))
    net(input, input)
    net.eval()
    net(input)

    pass
