"""Chunking:句子边界感知的固定大小切分(带 overlap)。

策略:先按中文句末标点/换行切成句子,再把句子顺序打包进 chunk,
装不下就封口并把上一块的尾部 overlap 个字符带入下一块(保持上下文连续);
单句超过 chunk_size 时硬切,避免超长块。
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


def chunk_text(text, chunk_size=256, overlap=32):
    """text -> List[str]。chunk_size/overlap 以字符计(中文场景字符≈token 量级)。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    overlap = max(0, min(overlap, chunk_size // 2))   # 防止 overlap 过大导致死循环

    sents = [s for s in _SENT_SPLIT.split(text) if s]
    chunks = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur)
        cur = cur[-overlap:] if overlap else ""

    for s in sents:
        if cur and len(cur) + len(s) > chunk_size:
            flush()
        cur += s
        while len(cur) > chunk_size:                  # 单句超长:硬切
            chunks.append(cur[:chunk_size])
            cur = cur[chunk_size - overlap:]
    if cur.strip() and (not chunks or cur != chunks[-1][-len(cur):]):
        chunks.append(cur)
    return chunks
