"""用训练好的 mini-GPT 生成唐诗:对比采样策略 + KV cache 开/关速度。

用法:python sample.py [--prompt 白日依山盡，] [--max_new 120]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent

STRATEGIES = [
    ("greedy",                 dict(temperature=0)),
    ("temp=0.8 top-k=50",      dict(temperature=0.8, top_k=50, top_p=None)),
    ("temp=0.8 top-p=0.9",     dict(temperature=0.8, top_p=0.9)),
    ("temp=1.2 top-p=0.95",    dict(temperature=1.2, top_p=0.95)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="白日依山盡，")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tok = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prompt_ids = tok.encode(args.prompt)

    print(f"prompt: {args.prompt}\n")
    print("=" * 20 + " 采样策略对比 " + "=" * 20)
    for name, kw in STRATEGIES:
        torch.manual_seed(args.seed)
        out = model.generate(prompt_ids, max_new_tokens=args.max_new, **kw)
        text = tok.decode(out)
        print(f"\n--- {name} ---\n{text}")

    print("\n" + "=" * 20 + " KV cache 速度对比 " + "=" * 20)
    for use_cache in (True, False):
        torch.manual_seed(args.seed)
        t0 = time.time()
        out = model.generate(prompt_ids, max_new_tokens=256,
                             temperature=0.8, top_p=0.9, use_cache=use_cache)
        dt = time.time() - t0
        print(f"use_cache={use_cache}: 256 tokens 用时 {dt:.2f}s "
              f"({256 / dt:.1f} tok/s)")


if __name__ == "__main__":
    main()
