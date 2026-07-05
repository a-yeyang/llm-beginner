"""Causal multi-head attention,支持 RoPE 与 KV cache。

与任务一的 encoder attention 的区别:
  1. causal mask:位置 i 只能看 j<=i(上三角置 -inf)
  2. RoPE 作用在 q/k 上(v 不旋转)
  3. KV cache:增量解码时把历史 k/v 缓存下来,每步只算新 token 的 q/k/v,
     注意新 token 的 RoPE offset 和 mask 都要按"全局位置"来
"""
import math

import torch
import torch.nn as nn

from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, kv_cache=None, return_cache=False):
        """x: (B, T, C);kv_cache: (k, v) 各 (B, H, past, hd) 或 None。"""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        past = kv_cache[0].size(2) if kv_cache is not None else 0
        q = apply_rope(q, cos, sin, offset=past)   # 新 token 的全局位置从 past 开始
        k = apply_rope(k, cos, sin, offset=past)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)  # (B, H, past+T, hd)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if return_cache else None

        S = k.size(2)                               # 总 key 长度 = past + T
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, S)
        if T > 1:
            # query i 的全局位置是 past+i,只能看 j <= past+i;
            # T==1(增量解码)时最后一个 token 本来就能看到全部缓存,无需 mask
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).triu(diagonal=past + 1)
            att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        if return_cache:
            return out, new_cache
        return out
