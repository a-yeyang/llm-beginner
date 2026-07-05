"""mini-GPT:decoder-only Transformer(Pre-LN + RoPE + KV cache)。

接口约定(README「实现约定」):
  - MiniGPT.forward(ids, kv_cache=None, return_cache=False)
  - MiniGPT.generate(prompt_ids, max_new_tokens, top_p, temperature)
  - 属性 block_size / max_seq_len(自检按它对困惑度切窗)
  - load_for_eval(ckpt_path) -> (model, tokenizer)

kv_cache 结构:list,每层一个 (k, v) 元组,k/v 形状 (B, H, past, head_dim)。
"""
from pathlib import Path

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .rope import rope_cos_sin
from .tokenizer import BPETokenizer


class Block(nn.Module):
    """Pre-LN decoder block:x + Attn(LN(x)),再 x + MLP(LN(x))。"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, kv_cache=None, return_cache=False):
        if return_cache:
            a, new_cache = self.attn(self.ln1(x), cos, sin, kv_cache, return_cache=True)
        else:
            a, new_cache = self.attn(self.ln1(x), cos, sin, kv_cache), None
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6,
                 block_size=256, dropout=0.1, rope_base=10000.0):
        super().__init__()
        self.block_size = block_size       # 训练上下文长度;自检按它切窗
        self.max_seq_len = block_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # 权重绑定,省参数且训练更稳

        # RoPE 表多留一倍余量:自检的滑窗喂 block_size+1 个 token,不应越界
        cos, sin = rope_cos_sin(2 * block_size, d_model // n_heads, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, kv_cache=None, return_cache=False):
        """ids: (B, T) -> logits (B, T, V);return_cache 时返回 (logits, new_kv_cache)。"""
        x = self.drop(self.embed(ids))
        new_cache = []
        for i, blk in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, c = blk(x, self.rope_cos, self.rope_sin, layer_cache, return_cache)
            new_cache.append(c)
        logits = self.head(self.ln_f(x))
        if return_cache:
            return logits, new_cache
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, top_p=0.9,
                 temperature=1.0, top_k=None, use_cache=True):
        """自回归生成。prompt_ids: List[int] 或 (T,)/(1,T) 张量;返回含 prompt 的 id 列表。"""
        from .sampling import sample_next

        self.eval()
        device = next(self.parameters()).device
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.view(1, -1).to(device)
        # 上下文上限:RoPE 表长度(2*block_size);超出则截最近的一段重建 cache
        ctx_limit = self.rope_cos.size(0)

        cache = None
        cur = ids
        for _ in range(max_new_tokens):
            if ids.size(1) >= ctx_limit:
                ids = ids[:, -self.block_size:]
                cache, cur = None, ids           # 截断后重新预填充
            if use_cache:
                logits, cache = self(cur, kv_cache=cache, return_cache=True)
            else:
                logits = self(ids)
            next_id = sample_next(logits[0, -1], temperature=temperature,
                                  top_k=top_k, top_p=top_p)
            nxt = torch.tensor([[next_id]], dtype=torch.long, device=device)
            ids = torch.cat([ids, nxt], dim=1)
            cur = nxt if use_cache else ids
        return ids[0].tolist()


def load_for_eval(ckpt_path):
    """从 checkpoint 还原 (model, tokenizer)。tokenizer.json 与 ckpt 同目录。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MiniGPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = BPETokenizer.from_pretrained(str(Path(ckpt_path).parent / "tokenizer.json"))
    return model, tok
