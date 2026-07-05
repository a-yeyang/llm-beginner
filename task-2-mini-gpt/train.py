"""任务二训练脚本:在唐诗语料上预训练 mini-GPT(next-token prediction)。

用法:
    python data/download.py                 # 或用更大的中文语料生成 train/dev.txt
    python train.py                         # 默认超参(GPU 数分钟)
    python train.py --steps 6000 --d_model 384

流程:训练(或加载)BPE tokenizer -> 编码语料(缓存到 data/*.ids.pt)->
随机窗口采样训练 -> 按 dev loss 保存 ckpt/best.pt -> 存训练历史供画图。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


def load_or_train_tokenizer(train_text, vocab_size, force=False):
    tok_path = ROOT / "ckpt" / "tokenizer.json"
    if tok_path.exists() and not force:
        tok = BPETokenizer.from_pretrained(str(tok_path))
        print(f"加载已有 tokenizer:vocab_size={tok.vocab_size}")
        return tok
    print(f"训练 BPE tokenizer(vocab_size={vocab_size})...")
    t0 = time.time()
    tok = BPETokenizer.train(train_text, vocab_size=vocab_size, verbose=True)
    tok_path.parent.mkdir(exist_ok=True)
    tok.save(str(tok_path))
    print(f"tokenizer 训练完成({time.time() - t0:.0f}s),已存 {tok_path.name}")
    return tok


def encode_cached(tok, text, cache_path, name):
    if cache_path.exists():
        ids = torch.load(cache_path, weights_only=True)
        print(f"加载缓存 {name}:{len(ids)} tokens")
        return ids
    print(f"编码 {name} 语料(纯 Python BPE,首次需几分钟)...")
    t0 = time.time()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save(ids, cache_path)
    ratio = len(text.encode('utf-8')) / max(1, len(ids))
    print(f"{name}: {len(ids)} tokens({time.time() - t0:.0f}s,压缩率 {ratio:.2f} 字节/token)")
    return ids


def get_batch(ids, block_size, batch_size, device):
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[i: i + block_size] for i in ix])
    y = torch.stack([ids[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_dev_loss(model, dev_ids, block_size, device, n_batches=20, batch_size=32):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(dev_ids, block_size, batch_size, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain_tokenizer", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    dev_text = (ROOT / "data" / "dev.txt").read_text(encoding="utf-8")

    tok = load_or_train_tokenizer(train_text, args.vocab_size, force=args.retrain_tokenizer)
    train_ids = encode_cached(tok, train_text, ROOT / "data" / "train.ids.pt", "train")
    dev_ids = encode_cached(tok, dev_text, ROOT / "data" / "dev.ids.pt", "dev")

    config = dict(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout)
    model = MiniGPT(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params / 1e6:.2f}M, vocab = {tok.vocab_size}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warmup = int(args.steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    best_dev = float("inf")
    history = {"step": [], "train_loss": [], "dev_loss": []}

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        sched.step()
        running += loss.item()

        if step % args.eval_interval == 0:
            train_loss = running / args.eval_interval
            running = 0.0
            dev_loss = eval_dev_loss(model, dev_ids, args.block_size, device)
            ppl = math.exp(dev_loss)
            speed = step / (time.time() - t0)
            print(f"step {step}/{args.steps} train {train_loss:.4f} "
                  f"dev {dev_loss:.4f} (ppl {ppl:.1f}) {speed:.1f} it/s")
            history["step"].append(step)
            history["train_loss"].append(round(train_loss, 4))
            history["dev_loss"].append(round(dev_loss, 4))
            if dev_loss < best_dev:
                best_dev = dev_loss
                torch.save({"state_dict": model.state_dict(), "config": config,
                            "step": step, "dev_loss": dev_loss}, ckpt_dir / "best.pt")
                print(f"  ✓ 保存最佳模型 (dev ppl {ppl:.1f}) -> ckpt/best.pt")

    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures" / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n训练结束,best dev loss {best_dev:.4f} (ppl {math.exp(best_dev):.1f})")


if __name__ == "__main__":
    main()
