"""任务四消融实验:chunk size 扫描 × reranker 开关,对 30 题 gold QA 报召回指标。

用法:python run_experiments.py
产出:figures/experiments.json + figures/ablation.png + 控制台表格
"""
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.embedding import BGEEmbedder
from src.indexer import build_index
from src.retriever import Retriever

CHUNK_SIZES = [128, 256, 512, 1024]


def normalize(t):
    return re.sub(r"\s+", "", str(t))


def load_gold():
    lines = (ROOT / "data" / "gold_qa.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def eval_recall(retriever, gold):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = []
    for item in gold:
        anchors = [normalize(a) for a in item["gold_anchors"]]
        results = retriever.retrieve(item["question"], k=10)
        rank = None
        for i, r in enumerate(results, 1):
            text = normalize(r["text"])
            if any(a and a in text for a in anchors):
                rank = i
                break
        if rank:
            for k in hits:
                if rank <= k:
                    hits[k] += 1
        rr.append(1 / rank if rank else 0)
    n = len(gold)
    m = {f"recall@{k}": round(v / n, 3) for k, v in hits.items()}
    m["mrr"] = round(sum(rr) / n, 3)
    return m


def main():
    gold = load_gold()
    embedder = BGEEmbedder()          # 各配置共用,省显存和加载时间
    results = []

    for cs in CHUNK_SIZES:
        overlap = cs // 4
        idx_dir = ROOT / "data" / (f"index" if cs == 256 else f"index_cs{cs}")
        if not (idx_dir / "faiss.index").exists():
            build_index(chunk_size=cs, overlap=overlap, out_dir=idx_dir,
                        embedder=embedder)
        for use_rr in (False, True):
            t0 = time.time()
            r = Retriever(index_dir=idx_dir, use_reranker=use_rr)
            r.embedder = embedder
            m = eval_recall(r, gold)
            m.update(chunk_size=cs, overlap=overlap, reranker=use_rr,
                     seconds=round(time.time() - t0, 1))
            results.append(m)
            print(f"cs={cs:4d} rerank={str(use_rr):5s} " +
                  " ".join(f"{k}={m[k]}" for k in
                           ["recall@1", "recall@5", "recall@10", "mrr"]))

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    (fig_dir / "experiments.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画图:recall@10 与 MRR 随 chunk size 变化,rerank 两条线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(["recall@10", "mrr"], axes):
        for use_rr, style in ((False, "o--"), (True, "s-")):
            ys = [m[metric] for m in results if m["reranker"] == use_rr]
            ax.plot(CHUNK_SIZES, ys, style,
                    label="rerank" if use_rr else "vector only")
        ax.set_xscale("log", base=2)
        ax.set_xticks(CHUNK_SIZES, [str(c) for c in CHUNK_SIZES])
        ax.set_xlabel("chunk size (chars)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("RAG ablation: chunk size × reranker (30 gold QA)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation.png", dpi=150)
    print(f"\n结果与图已写入 {fig_dir}")


if __name__ == "__main__":
    main()
