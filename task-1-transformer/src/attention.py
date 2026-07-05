"""任务一核心：从零手写 attention。

本文件不依赖 nn.MultiheadAttention 等任何高层封装，只用最基础的
matmul / softmax / linear 拼出：

1. scaled_dot_product_attention —— 缩放点积注意力（含 mask）
2. MultiHeadAttention            —— 多头自注意力

mask 约定（与 README「实现约定」、eval/run.py 一致）：
    mask 中 True 表示「该位置被屏蔽」，会被填成 -inf，softmax 后概率为 0。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, return_attn=False):
    """缩放点积注意力。

    Args:
        Q, K, V: 形状均为 (B, H, T, D_k)。B=batch, H=head 数, T=序列长, D_k=每个 head 的维度。
        mask:    可广播到 (B, H, T_q, T_k) 的布尔张量；True = 被屏蔽。
                 - padding mask 常见形状 (B, 1, 1, T_k)
                 - causal  mask 常见形状 (T_q, T_k) 的上三角
        return_attn: 为 True 时额外返回注意力权重（可视化用）。

    Returns:
        out: (B, H, T, D_k)。return_attn=True 时返回 (out, attn)，attn 形状 (B, H, T_q, T_k)。
    """
    d_k = Q.size(-1)

    # 1) 打分：Q·Kᵀ，再除以 sqrt(d_k)。
    #    注意缩放因子是 sqrt(d_k)（每个 head 的维度），不是 sqrt(d_model)。
    #    缩放是为了让点积方差稳定在 1 附近，否则维度一大 softmax 会进饱和区、梯度消失。
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T_q, T_k)

    # 2) mask：被屏蔽位置填 -inf。
    #    用 -inf 而不是「乘 0」——softmax(-inf)=0 才能真正不泄漏；乘 0 会让被屏蔽位置
    #    仍带着 exp(score) 的概率质量参与归一化。
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # 3) 在最后一维 (T_k，即「被注意的 key」那一维) 上做 softmax。
    #    每个 query 对所有 key 的注意力权重之和为 1。
    attn = F.softmax(scores, dim=-1)

    # 4) 用注意力权重对 V 加权求和。
    out = torch.matmul(attn, V)  # (B, H, T_q, D_k)

    if return_attn:
        return out, attn
    return out


class MultiHeadAttention(nn.Module):
    """多头自注意力。

    把 d_model 维拆成 H 个 head，每个 head 独立做缩放点积注意力，再拼回来过一个输出投影。
    多头让模型在不同子空间里关注不同的关系（语法 / 语义 / 位置 …）。
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q/K/V 各自独立的投影矩阵（不要偷懒共用一个）。输出再单独投影一次。
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, H, T, d_k)。

        关键是 transpose 顺序：先 view 成 (B, T, H, d_k)，再把 H 换到第 2 维，
        得到 (B, H, T, d_k)，这样每个 head 在 T 维上独立做 attention。
        """
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, H, T, d_k) -> (B, T, d_model)。

        transpose 之后内存不连续，必须 .contiguous() 才能 view，否则报错。
        """
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x, mask=None, return_attn=False):
        """自注意力：Q/K/V 都来自同一个输入 x。

        Args:
            x:    (B, T, d_model)
            mask: 可广播到 (B, H, T, T) 的布尔张量，True = 屏蔽。
        """
        Q = self._split_heads(self.W_q(x))  # (B, H, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        if return_attn:
            ctx, attn = scaled_dot_product_attention(Q, K, V, mask=mask, return_attn=True)
        else:
            ctx = scaled_dot_product_attention(Q, K, V, mask=mask)

        out = self.W_o(self._merge_heads(ctx))  # (B, T, d_model)
        out = self.dropout(out)

        if return_attn:
            return out, attn
        return out
