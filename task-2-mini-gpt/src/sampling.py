"""解码/采样策略:greedy / temperature / top-k / top-p。

输入约定:logits 为最后一个位置的未归一化分数,形状 (V,)。
"""
import torch


def greedy(logits):
    return int(logits.argmax(dim=-1).item())


def top_k_filter(logits, k):
    """只保留分数最高的 k 个 token,其余置 -inf。"""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k).values[..., -1]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """核采样:按概率从大到小累加,保留累计概率首次达到 p 的最小集合。"""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # 右移一位:保证至少保留概率最高的一个 token
    cut = cum - probs > p
    sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """组合策略采样下一个 token id。temperature<=0 退化为 greedy。"""
    if temperature is None or temperature <= 0:
        return greedy(logits)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
