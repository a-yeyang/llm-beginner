"""检索器:向量召回(可选 reranker 精排)。

Retriever() 无参构造即可用(自检约定):索引不存在时自动从 kb.pdf 构建。
默认两段式:FAISS 粗排召回 rerank_candidates 个 -> bge-reranker 精排取 top-k。
"""
import json
from pathlib import Path

import faiss

from .embedding import BGEEmbedder

ROOT = Path(__file__).resolve().parents[1]


class Retriever:
    def __init__(self, index_dir=None, use_reranker=True, rerank_candidates=30):
        self.index_dir = Path(index_dir or ROOT / "data" / "index")
        if not (self.index_dir / "faiss.index").exists():
            from .indexer import build_index
            print(f"索引不存在,从 kb.pdf 构建 -> {self.index_dir}")
            build_index(out_dir=self.index_dir)
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.records = json.loads(
            (self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.embedder = BGEEmbedder()
        self.rerank_candidates = rerank_candidates
        self._reranker = None
        self.use_reranker = use_reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from .reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def retrieve(self, query, k=5):
        """query -> List[dict(text, score, source)],按相关性降序。"""
        n_recall = max(k, self.rerank_candidates if self.use_reranker else k)
        emb = self.embedder.encode([query], is_query=True).numpy()
        scores, idx = self.index.search(emb, n_recall)
        results = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            r = dict(self.records[i])
            r["score"] = float(s)
            results.append(r)
        if self.use_reranker:
            results = self.reranker.rerank(query, results, k=k)
        return results[:k]
