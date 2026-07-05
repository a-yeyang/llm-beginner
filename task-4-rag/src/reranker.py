"""bge-reranker-base 精排:cross-encoder,对 (query, passage) 成对打分。

与向量检索(bi-encoder)的区别:reranker 让 query 和 passage 在注意力里
充分交互,精度高但代价是每对都要过一遍模型,只能用于粗排后的小候选集。
"""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "bge-reranker-base"


class Reranker:
    def __init__(self, model_dir=None, device=None):
        model_dir = str(model_dir or DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(self.device).eval()

    @torch.no_grad()
    def score(self, query, passages, batch_size=16, max_length=512):
        """返回每个 passage 的相关性分数 List[float](越大越相关)。"""
        scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i: i + batch_size]
            enc = self.tokenizer([query] * len(batch), batch,
                                 padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
        return scores

    def rerank(self, query, candidates, k=None):
        """candidates: List[dict(text=...)];返回按精排分数排序的前 k 个(分数写回 score)。"""
        scores = self.score(query, [c["text"] for c in candidates])
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return ranked[:k] if k else ranked
