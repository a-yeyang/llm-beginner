"""decoder-only 语言模型（mini-GPT）。复用任务一的 TransformerBlock 与位置编码。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .model import PositionalEncoding


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 d_ff=None, max_len=128, dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # 权重共享

        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self.apply(self._init_weights)  # GPT-2 风格初始化

    @staticmethod
    def _init_weights(m):
        """让初始 logits 接近均匀分布，step0 的 loss 就≈ln(vocab)（纯随机猜的基线）。"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attn=False, label_smoothing=0.0):
        B, T = idx.shape
        assert T <= self.max_len, f"序列 {T} 超过上下文窗口 {self.max_len}"
        mask = self.causal_mask[:T, :T]

        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        x = self.drop(self.pos(x))
        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, mask=mask, return_attn=True)
                attns.append(a)
            else:
                x = blk(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 训练时可开 label_smoothing 抑制过度自信；eval 用默认 0，保证 val loss 可比
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100,
                                   label_smoothing=label_smoothing)
        if return_attn:
            return logits, loss, attns
        return logits, loss

    def configure_optimizer(self, lr, weight_decay):
        """只对 2D 权重做 weight decay，LayerNorm/bias/嵌入不做（GPT 标准做法）。"""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
