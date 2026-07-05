"""BGE embedding:直接用 transformers 加载(不依赖 sentence-transformers)。

bge 系列约定:CLS pooling + L2 归一化;检索场景 query 侧加指令前缀,文档侧不加。
"""
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    def __init__(self, model_dir=None, device=None):
        model_dir = str(model_dir or DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=64, is_query=False, max_length=512):
        """List[str] -> (N, D) 归一化后的 float32 张量(CPU)。"""
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state[:, 0]   # CLS pooling
            outs.append(torch.nn.functional.normalize(hidden, dim=-1).cpu())
        return torch.cat(outs).float()
