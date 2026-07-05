"""RoPE(旋转位置编码)手写实现。

核心思想:不往 embedding 里加位置向量,而是把每个 head 维度两两配对成复平面上的点,
按位置 m 旋转角度 m·θ_i(θ_i = base^(-2i/d))。这样 q·k 的内积只依赖相对位置 m-n,
天然具备相对位置感知,且推理时可以只对新 token 旋转(配合 KV cache)。
"""
import torch


def rope_cos_sin(seq_len, head_dim, base=10000.0, device=None, dtype=torch.float32):
    """预计算 cos/sin 表,各为 (seq_len, head_dim/2)。"""
    assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # (T, D/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin, offset=0):
    """对 q 或 k 应用旋转。x: (B, H, T, D);offset = 该段序列的起始绝对位置。

    KV cache 增量解码时,新 token 的绝对位置不是 0 而是"已缓存长度",
    必须用 offset 取对应位置的 cos/sin,否则增量结果与全量不一致。
    """
    T, D = x.size(2), x.size(3)
    c = cos[offset: offset + T].view(1, 1, T, D // 2)
    s = sin[offset: offset + T].view(1, 1, T, D // 2)
    x1, x2 = x[..., 0::2], x[..., 1::2]              # 偶/奇维配对
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s                 # 复数乘法的实部/虚部
    out[..., 1::2] = x1 * s + x2 * c
    return out
