"""任务一训练脚本：在 ChnSentiCorp 上训练字符级 Transformer 文本分类器。

用法：
    python data/download.py        # 先下数据
    python train.py                # 默认超参直接训
    python train.py --epochs 8 --d_model 128 --n_layers 4 --lr 3e-4

训练完会把最佳模型存到 ckpt/best.pt（含 state_dict / config / vocab），
之后 python eval/run.py 的 classifier_accuracy 一项就能跑。
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

# Windows 控制台默认 CP936，打印 ✓ 等字符会 UnicodeEncodeError（与 _eval_harness 同款处理）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN, TransformerClassifier

ROOT = Path(__file__).resolve().parent


# ----------------------------- 数据 -----------------------------

def build_vocab(texts, min_freq=1):
    """从训练文本构建字符级词表。0=<pad>, 1=<unk>，其余按词频排。"""
    counter = Counter()
    for t in texts:
        counter.update(str(t))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for ch, freq in counter.most_common():
        if freq >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


class ChnSentiDataset(Dataset):
    def __init__(self, df, vocab, max_len=256):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = [self.vocab.get(ch, UNK_ID) for ch in str(self.texts[i])[: self.max_len]]
        if not ids:
            ids = [PAD_ID]
        return torch.tensor(ids, dtype=torch.long), int(self.labels[i])


def collate(batch):
    """把不等长序列右侧补 PAD 到 batch 内最大长度。"""
    seqs, labels = zip(*batch)
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out, torch.tensor(labels, dtype=torch.long)


# ----------------------------- 训练 -----------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        pred = model(ids).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    print(f"train = {len(train_df)}, dev = {len(dev_df)}")

    vocab = build_vocab(train_df["text"])
    print(f"vocab size = {len(vocab)}")

    train_ds = ChnSentiDataset(train_df, vocab, args.max_len)
    dev_ds = ChnSentiDataset(dev_df, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)

    config = dict(
        vocab_size=len(vocab), d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, num_classes=2, max_len=args.max_len,
        dropout=args.dropout, pad_id=PAD_ID,
    )
    model = TransformerClassifier(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        # 线性 warmup + cosine 衰减
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (ids, labels) in enumerate(train_loader, 1):
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            sched.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running/step:.4f} lr {sched.get_last_lr()[0]:.2e}")

        acc = evaluate(model, dev_loader, device)
        print(f"[epoch {epoch}] dev_acc = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "vocab": vocab, "unk_id": UNK_ID}, ckpt_dir / "best.pt")
            print(f"  ✓ 保存最佳模型 (dev_acc={acc:.4f}) -> ckpt/best.pt")

    print(f"\n训练结束，最佳 dev_acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
