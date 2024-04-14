# -*- coding: utf-8 -*-
"""
@Time ： 3/22/24 2:24 AM
@Auth ： woldier wong
@File ：window_transformer.py
@IDE ：PyCharm
@DESCRIPTION：窗口注意力机制
"""
import torch, torch.nn as nn, torch.nn.functional as F
from model.our.attention import Attention, WindowAttention, Mlp, LiPeFFN
from timm.models.layers import DropPath, to_2tuple
import math


class WindowTransformerBlock(nn.Module):
    """
    窗口transformer
    >>> net = WindowTransformerBlock(
    >>>     dim=64,
    >>>     input_resolution=(256, 256),
    >>>     num_heads=8,
    >>>     token_mlp='LP'
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
            input_resolution,
            num_heads: int,
            win_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale=None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            token_projection: str = 'linear',
            token_mlp: str = 'LiPe',
            modulator: bool = False,
            cross_modulator: bool = False
    ):
        """

        :param dim: 输入的dim
        :param input_resolution: 切分窗口前的图片大小
        :param num_heads: 多头的个数
        :param win_size: 窗口大小
        :param shift_size: shift 的大小, 模仿swin trans shifted-window
        :param mlp_ratio: mlp 中 隐藏层是原始输入的几倍
        :param qkv_bias:
        :param qk_scale:
        :param drop: 投影层的drop
        :param attn_drop: atten中的drop
        :param drop_path: atten后与mpl连接时的drop
        :param act_layer: 激活函数
        :param norm_layer: norm 层
        :param token_projection: 线性投影的类型, 可选 "linear"和"conv"
        :param token_mlp: FFN 的类型,   可选 "mlp"和"LiPe"
        :param modulator:  是否需要调制器
        :param cross_modulator:  是否需要cross 调制器
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
            self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                        proj_drop=drop,
                                        token_projection=token_projection, )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None
        # 第一个norm layer
        self.norm1 = norm_layer(dim)
        # window attn
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        # Drop paths (Stochastic Depth) per sample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个norm layer
        self.norm2 = norm_layer(dim)
        # 计算mlp 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'LiPe':
            self.mlp = LiPeFFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            # x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x_cross = self.cross_attn(x_cross, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # 转成图片

        # cyclic shift
        shifted_x = self._shfit_window(x)

        # partition windows, 切分成window
        x_windows = self._partition_windows(shifted_x, C)

        # with_modulator
        wmsa_in = self._with_modulator(x_windows)

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in)  # nW*B, win_size*win_size, C

        # merge windows  # 将window 合成图片
        shifted_x = self._merge_windows(attn_windows, C, H, W)

        # reverse cyclic shift  # 如果是经过了shift, 再次进行de shift
        x = self._reverse_shift_window(shifted_x, B, C, H, W)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _reverse_shift_window(self, shifted_x, B, C, H, W):
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        return x

    def _merge_windows(self, attn_windows, C, H, W):
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
        return shifted_x

    def _with_modulator(self, x_windows):
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows
        return wmsa_in

    def _partition_windows(self, shifted_x, C):
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        return x_windows

    def _shfit_window(self, x):
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        return shifted_x


# =======================================
# =============image to window===========
# =======================================
def window_partition(x, win_size, dilation_rate=1):
    """
    >>> feature_map = torch.randn((8, 64, 128, 128))
    >>> feature_map_c_last = feature_map.permute(0, 2, 3, 1)
    >>> window_map = window_partition(feature_map_c_last, 4)
    >>> window_map_reshape = window_map.view(-1, 4 * 4, 64)
    >>> # 送入计算attn
    >>> feature_map_reverse = window_reverse(window_map_reshape.view(-1, 4, 4, 64), 4, 128, 128)
    >>> print(torch.allclose(feature_map_c_last, feature_map_reverse))

    :param x:
    :param win_size:
    :param dilation_rate:
    :return:
    """
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


if __name__ == '__main__':
    net = WindowTransformerBlock(
        dim=64,
        input_resolution=(256, 256),
        num_heads=8,
        token_projection="conv"
    )
    # 一个特征图 [B, H, W, C]
    feature_map = torch.randn((1, 256, 256, 64))
    shape = feature_map.shape
    # 转换[B, H*W, C]
    feature_map = feature_map.view(shape[0], -1, shape[-1])
    out = net(feature_map)
    pass
