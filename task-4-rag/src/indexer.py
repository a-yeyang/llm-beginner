"""索引构建:kb.pdf -> 文本提取 -> chunking -> BGE embedding -> FAISS。

索引产物存 data/index*/ 三件套:faiss.index、chunks.json(文本+来源页)、meta.json。
用内积索引(IndexFlatIP)+ 归一化向量 = 余弦相似度;精确检索,千级 chunk 无需近似。
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from .chunker import chunk_text
from .embedding import BGEEmbedder

ROOT = Path(__file__).resolve().parents[1]


def extract_pdf_text(pdf_path):
    """逐页提取文本,返回 (全文, 每页在全文中的起始偏移) 用于溯源。"""
    reader = PdfReader(str(pdf_path))
    pages, offsets, pos = [], [], 0
    for page in reader.pages:
        t = page.extract_text() or ""
        t = re.sub(r"[ \t]+", " ", t)          # 收敛 PDF 提取的碎空格,保留换行给切句用
        pages.append(t)
        offsets.append(pos)
        pos += len(t)
    return "".join(pages), offsets


def _page_of(offset, page_offsets):
    """二分查偏移所在页(1-based)。"""
    lo, hi = 0, len(page_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if page_offsets[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def build_index(chunk_size=256, overlap=64, out_dir=None,
                pdf_path=None, embedder=None, verbose=True):
    """构建并落盘索引,返回 out_dir。"""
    pdf_path = pdf_path or ROOT / "data" / "kb.pdf"
    out_dir = Path(out_dir or ROOT / "data" / "index")
    out_dir.mkdir(parents=True, exist_ok=True)

    text, page_offsets = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if verbose:
        print(f"PDF {len(page_offsets)} 页,{len(text)} 字符 -> {len(chunks)} chunks "
              f"(size={chunk_size}, overlap={overlap})")

    # 打包是顺序进行的,用累计步长估算每块起始偏移 -> 页码(溯源展示用,允许近似)
    records, pos = [], 0
    for c in chunks:
        records.append({"text": c, "source": f"kb.pdf p.{_page_of(pos, page_offsets)}"})
        pos += max(1, len(c) - overlap)

    embedder = embedder or BGEEmbedder()
    emb = embedder.encode([r["text"] for r in records]).numpy().astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "chunk_size": chunk_size, "overlap": overlap,
        "n_chunks": len(records), "dim": int(emb.shape[1]),
        "embedding_model": "bge-small-zh-v1.5",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"索引已写入 {out_dir}")
    return out_dir
