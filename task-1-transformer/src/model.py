"""任务一模型：堆叠 encoder block 的文本分类器，以及评测用的 load_for_eval 工厂。

接口约定（见 README「实现约定」）：
    - class TransformerClassifier：forward(ids) -> logits (B, num_classes)
    - load_for_eval(ckpt_path) -> (model, tokenize_fn)
      tokenize_fn(text) -> LongTensor 形状 (T,)，评测里会 .unsqueeze(0) 成 batch。

分词采用「字符级」：对中文最简单稳健，且完全离线、无需下载任何预训练 tokenizer。
词表在训练时构建并存进 checkpoint，load_for_eval 时还原。
"""
import math

import torch
import torch.nn as nn

from .block import TransformerBlock

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class PositionalEncoding(nn.Module):
    """正弦位置编码。

    attention 本身对位置「无感」（打乱 token 顺序结果不变），所以要显式注入位置信息。
    这里用原论文的 sin/cos 绝对位置编码，注册成 buffer（随模型保存、但不训练）。
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()           # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))             # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)，按实际序列长截取位置编码相加
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """词嵌入 + 位置编码 + N 层 encoder block + 池化 + 分类头。"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, num_classes=2, max_len=512, dropout=0.1, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)          # 最后再 LN 一次（Pre-LN 网络的惯例）
        self.head = nn.Linear(d_model, num_classes)

    def _padding_mask(self, ids):
        """ids (B, T) -> mask (B, 1, 1, T)，True = PAD（被屏蔽）。

        形状 (B,1,1,T) 能广播到注意力打分 (B, H, T, T)：对每个 query 屏蔽掉所有 PAD key。
        """
        return (ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def forward(self, ids, return_attn=False):
        """Args: ids (B, T) 的 token id。Returns: logits (B, num_classes)。"""
        mask = self._padding_mask(ids)

        # 词嵌入乘 sqrt(d_model) 是原论文的缩放惯例，让 embedding 与位置编码量级匹配。
        x = self.embed(ids) * math.sqrt(self.d_model)
        x = self.dropout(self.pos(x))

        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, mask=mask, return_attn=True)
                attns.append(a)
            else:
                x = blk(x, mask=mask)
        x = self.norm(x)

        # 池化：对非 PAD 位置做 masked mean（务必排除 PAD，否则被填充位置会拉偏句向量）。
        keep = (ids != self.pad_id).unsqueeze(-1).float()     # (B, T, 1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        logits = self.head(pooled)                            # (B, num_classes)
        if return_attn:
            return logits, attns
        return logits


def build_tokenize_fn(vocab, max_len=512, pad_id=PAD_ID, unk_id=UNK_ID):
    """根据词表生成 tokenize 函数：text -> LongTensor (T,)。"""
    def tokenize_fn(text):
        ids = [vocab.get(ch, unk_id) for ch in str(text)[:max_len]]
        if not ids:                       # 空串兜底，避免 0 长度序列
            ids = [pad_id]
        return torch.tensor(ids, dtype=torch.long)
    return tokenize_fn


def load_for_eval(ckpt_path):
    """评测工厂：从 checkpoint 还原 (model, tokenize_fn)。

    checkpoint 是一个 dict，含：state_dict / config / vocab，由 train.py 保存。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["vocab"]

    model = TransformerClassifier(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenize_fn = build_tokenize_fn(
        vocab,
        max_len=cfg.get("max_len", 512),
        pad_id=cfg.get("pad_id", PAD_ID),
        unk_id=ckpt.get("unk_id", UNK_ID),
    )
    return model, tokenize_fn
