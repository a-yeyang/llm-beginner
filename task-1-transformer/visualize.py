"""任务一 M5：画注意力热图。

用法（先训练出 ckpt/best.pt）：
    python visualize.py --layer 0 --head 0

会读 ckpt/best.pt，对几个示例句子前向一遍，把指定 layer/head 的注意力矩阵
(T, T) 用 imshow 画出来，存到 figures/。x/y 轴标 token（字）。
"""
import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent

# 示例句子：1 正面、1 负面、1 长句
SENTENCES = [
    "这家酒店的服务态度非常好，房间也很干净，下次还会再来。",
    "送货太慢了，客服态度也很差，非常失望，再也不会买了。",
    "整体来说性价比还可以，外观设计不错，但是电池续航一般，发热也比较明显，希望厂家后续能改进。",
]


def pick_cjk_font():
    """尽量挑一个能显示中文的字体，挑不到就算了（图仍能出，中文可能显示成方块）。"""
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


def plot_one(attn, tokens, title, out_path):
    # attn: (T, T) numpy；tokens: list[str]
    fig, ax = plt.subplots(figsize=(0.45 * len(tokens) + 2, 0.45 * len(tokens) + 2))
    im = ax.imshow(attn, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key (被注意的位置)")
    ax.set_ylabel("Query (发起注意的位置)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0, help="取第几层 block 的注意力")
    ap.add_argument("--head", type=int, default=0, help="取第几个 head")
    args = ap.parse_args()

    pick_cjk_font()
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise SystemExit("ckpt/best.pt 不存在，请先 python train.py")

    model, tokenize_fn = load_for_eval(str(ckpt))
    model.eval()

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, sent in enumerate(SENTENCES):
            ids = tokenize_fn(sent)
            logits, attns = model(ids.unsqueeze(0), return_attn=True)
            pred = int(logits.argmax(-1))
            # attns[layer]: (B, H, T, T) -> 取 batch0、指定 head
            attn = attns[args.layer][0, args.head].cpu().numpy()
            tokens = list(sent)[: ids.size(0)]
            label = "正面" if pred == 1 else "负面"
            title = f"layer{args.layer} head{args.head} | 预测：{label}"
            plot_one(attn, tokens, title, fig_dir / f"attn_{i}_L{args.layer}H{args.head}.png")

    print("\n完成。热图在 figures/ 下。")


if __name__ == "__main__":
    main()
