"""Transformer encoder block = 多头注意力 + 前馈网络 + 两个残差 + 两个 LayerNorm。

本实现采用 Pre-LN（LayerNorm 放在子层之前）：
    x = x + Sublayer(LayerNorm(x))
Pre-LN 比原始论文的 Post-LN 更好训练（梯度更稳，常常不需要 warmup 也能收敛），
是当下主流写法。两种都对，但全网络要保持一致。
"""
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """逐位置前馈网络：两层线性 + 中间激活，对每个 token 独立做一次非线性变换。

    惯例中间维 d_ff = 4 * d_model。激活用 GELU（比 ReLU 在 Transformer 里略好）。
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """一层 encoder block（Pre-LN）。"""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        """Args: x (B, T, d_model); mask 可广播到 (B, H, T, T)，True=屏蔽。"""
        # 子层一：注意力。先 LN 再进 attention，输出 dropout 后加回残差。
        if return_attn:
            a, attn = self.attn(self.ln1(x), mask=mask, return_attn=True)
        else:
            a = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(a)

        # 子层二：前馈网络。同样 Pre-LN + 残差。
        x = x + self.ffn(self.ln2(x))

        if return_attn:
            return x, attn
        return x
