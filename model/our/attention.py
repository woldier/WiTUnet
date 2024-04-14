# -*- coding: utf-8 -*-
"""
@Time ： 3/20/24 6:07 AM
@Auth ： woldier wong
@File ：attention.py
@IDE ：PyCharm
@DESCRIPTION：注意力机制
"""
import torch, torch.nn as nn
from typing import Optional
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math


# =======================================
# ===========Attention===================
# =======================================

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
            proj_drop=0.):
        """

        :param dim: 输入的序列通道数(C)或者维度数(D) [B, L, C] or [B, L, D]
        :param num_heads: multi-head 的数量
        :param token_projection:
        :param qkv_bias: qkv 的偏置值
        :param qk_scale:
        :param attn_drop: 注意力中 drop out 的比率
        :param proj_drop: 经过linear后的drop比率
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA)

    Instead of using global self-attention like the vanilla Transformer,
    we perform the self-attention within non-overlapping local windows,
    which reduces the computational cost significantly.
    Given the 2D feature maps X ∈ RC×H×W with H and W being the height and width of the maps,
    we split X into non-overlapping windows with the window size of M×M,
    and then get the flattened and transposed featuresXi ∈ RM2×C from each window i. Next, we perf

    the dimension of each head in Transformer block dk equals C.
    """

    def __init__(self,
                 dim: int,
                 win_size: tuple = (4, 4),
                 num_heads: int = 8,
                 token_projection: str = 'linear',
                 qkv_bias: bool = True,
                 qk_scale: torch.Tensor = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.
                 ):
        """

        :param dim: 输入维度
        :param win_size: 窗口的大小
        :param num_heads: 多头的个数
        :param token_projection: 怎么对input 进行编码得到qkv 可选 'linear' 'conv'
        :param qkv_bias: 是否需要在QK 之后加入bais
        :param qk_scale:
        :param attn_drop:
        :param proj_drop:
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 定义相对位置bias的可学习参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)  # 相对位置编码

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape  # 得到 输入的shape
        q, k, v = self.qkv(x, attn_kv)  # 计算QKV
        q = q * self.scale  # q进行scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =======================================
# =============QKV project===============
# =======================================

class LinearProjection(nn.Module):
    """
    线性投影层, 用于将[B,L,D] 投影到 [B,L,D'] head次
    """

    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.,
            bias: bool = True
    ):
        """

        :param dim: 输入sequence的隐藏维度(D)
        :param heads: 线性投影的次数 (heads次)
        :param dim_head: 线性投影后的隐藏层维度(D')
        :param dropout: dropout 比率
        :param bias: 对于线性投影中用到的
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(
            self,
            x: torch.Tensor,
            attn_kv=None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        # [B,N,C] -> [B, N, heads * dim_head] -> [B, N, 1, heads, dim_head] ->  [1, N, heads, N, dim_head]
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class ConvProjection(nn.Module):
    """
    >>> net = ConvProjection(64)
    >>> input = torch.randn((1024, 64, 64))
    >>> q,k,v = net(input)
    >>> print(q.shape)
    """
    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            kernel_size: int = 3,
            q_stride: int = 1,
            k_stride: int = 1,
            v_stride: int = 1,
            bias: bool = True
    ):
        """

        :param dim: 输入的维度
        :param heads: 多头数量
        :param dim_head: 每个头的维度
        :param kernel_size: 卷积核大小
        :param q_stride: q 步长
        :param k_stride: k 步长
        :param v_stride: v 步长
        :param bias:
        """
        super(ConvProjection,self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias=bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias=bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias=bias)

    def forward(self, x, attn_kv=None):
        # b: batch_size*window_num
        # n: 一个窗口内的像素值 Wh * Ww
        # c: 小窗口的通道数
        # h: 多头的个数
        b, n, c, h = *x.shape, self.heads
        # l = int(math.sqrt(n))
        l = w = int(math.sqrt(n))
        # 为了便于做cross-attn
        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)  # 变换成小窗口
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)  # 变换成小窗口
        # print(attn_kv)
        q = self.to_q(x)  # 计算Q
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)  # 变换回 [B, H, L, D]

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class SepConv2d(nn.Module):
    """
    >>> net = SepConv2d(64, 512,padding=1)
    >>> input = torch.randn((1024, 64, 8, 8))
    >>> print(net(input).shape)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool=True,
                 act_layer=nn.ReLU,
                 ):
        super(SepConv2d, self).__init__()
        # 通道数不变,3*3卷积范围上的Conv 更好的融合周围像素点的特征
        self.depthwise = nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels,
                                         bias=bias
                                   )
        # 通道数变化到目标通道, 卷积核为1, 做点式的emb
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


# =======================================
# =============feed-forward==============
# =======================================
class Mlp(nn.Module):
    """
    atten 之后的线性投影层
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            act_layer=nn.GELU,
            drop: float = 0.
    ):
        """
        MLP
        :param in_features: 输入的特征维度
        :param hidden_features: 隐藏层维度
        :param out_features: 输出层维度
        :param act_layer: 激活函数
        :param drop: drop比率
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ECALayer1D(nn.Module):
    """Constructs a ECA module.
    有效通道注意力模块
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECALayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class LiPeFFN(nn.Module):
    """
    Localized image  Perception enhancement FFN
    改进的窗口局部
    """

    def __init__(
            self,
            dim: int = 32,
            hidden_dim: int = 128,
            act_layer=nn.GELU,
            drop: float = 0.,
            use_eca: bool = False
    ):
        """

        :param dim: 输入维度
        :param hidden_dim: 隐藏层维度
        :param act_layer: 激活函数
        :param drop:
        :param use_eca:是否需要通道注意力
        """
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = ECALayer1D(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)  # 转换为图片
        # bs,hidden_dim,32x32

        x = self.dwconv(x)  # 做LiPe

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


if __name__ == '__main__':
    # net = SepConv2d(64, 512, 3, padding=1)
    # input = torch.randn((1024, 64, 8, 8))
    # print(net(input).shape)

    # net = ConvProjection(64)
    # input = torch.randn((1024, 64, 64))
    # q,k,v = net(input)
    # print(q.shape)
    pass
