"""训练 mini-GPT 并「看见」内部变化：loss/困惑度、梯度范数、各参数偏离初始值的程度，
以及训练前后注意力图的对比。

用法：python demo_gpt_train.py
产物：figures/gpt_loss.png、figures/gpt_param_evolution.png、figures/gpt_attn_before_after.png
"""
import json
import math
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

from src.gpt import MiniGPT

ROOT = Path(__file__).resolve().parent

# ---- 超参（小而快，CPU 几分钟）----
BLOCK = 64
BATCH = 32
MAX_STEPS = 2500
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
LR = 3e-4
D_MODEL, N_HEADS, N_LAYERS = 128, 4, 3
SEED = 1337


def pick_cjk_font():
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pick_cjk_font()

    # ---------- 数据 ----------
    text = (ROOT.parent / "poetryFromTang.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"device={device}  语料={len(data)} 字  词表={len(chars)} 个字  "
          f"train={len(train_data)}  val={len(val_data)}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
        x = torch.stack([d[i:i + BLOCK] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix]).to(device)
        return x, y

    # ---------- 模型 ----------
    model = MiniGPT(len(chars), D_MODEL, N_HEADS, N_LAYERS, max_len=BLOCK).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 = {n_params/1e6:.3f} M\n")

    # 监控这几组有代表性的参数：词嵌入 / 第0层 Q 投影 / 第0层 FFN / 第0层 LayerNorm
    tracked = {
        "tok_emb(词嵌入)": model.tok_emb.weight,
        "L0.W_q(查询投影)": model.blocks[0].attn.W_q.weight,
        "L0.ffn.fc1(前馈)": model.blocks[0].ffn.fc1.weight,
        "L0.ln1.gamma(层norm)": model.blocks[0].ln1.weight,
    }
    init_snapshot = {k: v.detach().clone() for k, v in tracked.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    @torch.no_grad()
    def estimate_loss(split):
        model.eval()
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        model.train()
        return losses.mean().item()

    @torch.no_grad()
    def sample(n_tokens=80, temperature=0.8, top_k=20):
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(ctx, n_tokens, temperature, top_k)[0].tolist()
        return "".join(itos[i] for i in out).replace("\n", " ")

    @torch.no_grad()
    def grab_attention(prompt="巴山上峡重复重"):
        model.eval()
        ids = torch.tensor([[stoi[c] for c in prompt if c in stoi]], device=device)
        _, _, attns = model(ids, return_attn=True)
        model.train()
        return attns[0][0, 0].cpu().numpy(), list(prompt)  # 第0层第0头 (T,T)

    # 训练前先抓一张注意力图（随机初始化状态）
    attn_before, attn_tokens = grab_attention()

    # ---------- 训练循环 + 监控 ----------
    history = {"step": [], "train": [], "val": [], "grad_norm": [], "ppl": [],
               "samples": [],
               "drift": {k: [] for k in tracked}, "wnorm": {k: [] for k in tracked},
               "meta": {"device": device, "n_chars": len(data), "vocab": len(chars),
                        "n_params": n_params, "d_model": D_MODEL, "n_heads": N_HEADS,
                        "n_layers": N_LAYERS, "block": BLOCK, "batch": BATCH,
                        "max_steps": MAX_STEPS, "lr": LR}}

    print(f"{'step':>5} | {'train':>6} {'val':>6} | {'困惑度PPL':>8} | {'梯度范数':>8} | 生成样本")
    print("-" * 100)

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            tr, va = estimate_loss("train"), estimate_loss("val")
            # 梯度范数：用一个 batch 反传一次量一下（不更新）
            x, y = get_batch("train")
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = math.sqrt(sum((p.grad.detach() ** 2).sum().item()
                                  for p in model.parameters() if p.grad is not None))
            opt.zero_grad(set_to_none=True)

            smp = sample(60)
            ppl = math.exp(min(tr, 20))  # 防溢出
            history["step"].append(step)
            history["train"].append(tr)
            history["val"].append(va)
            history["grad_norm"].append(gnorm)
            history["ppl"].append(ppl)
            history["samples"].append(smp)
            for k, v in tracked.items():
                drift = (v.detach() - init_snapshot[k]).norm().item()
                history["drift"][k].append(drift)
                history["wnorm"][k].append(v.detach().norm().item())

            print(f"{step:>5} | {tr:>6.3f} {va:>6.3f} | {ppl:>8.1f} | {gnorm:>8.2f} | {smp}")

        if step == MAX_STEPS:
            break
        # 正常训练 step
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    attn_after, _ = grab_attention()

    # ---------- 出图 ----------
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    steps = history["step"]

    # 1) loss 曲线
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, history["train"], "-o", label="train loss", ms=3)
    plt.plot(steps, history["val"], "-s", label="val loss", ms=3)
    plt.xlabel("training step"); plt.ylabel("cross-entropy loss")
    plt.title("mini-GPT loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fig_dir / "gpt_loss.png", dpi=140); plt.close()

    # 2) 参数演化：左=各参数偏离初始的相对幅度，右=全局梯度范数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in tracked:
        rel = [d / (init_snapshot[k].norm().item() + 1e-9) for d in history["drift"][k]]
        ax1.plot(steps, rel, "-o", ms=3, label=k)
    ax1.set_xlabel("training step"); ax1.set_ylabel("||W - W_init|| / ||W_init||")
    ax1.set_title("每组参数偏离随机初始的相对幅度"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(steps, history["grad_norm"], "-o", ms=3, color="crimson")
    ax2.set_xlabel("training step"); ax2.set_ylabel("global grad L2 norm")
    ax2.set_title("全局梯度范数（学习信号强度）"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "gpt_param_evolution.png", dpi=140); plt.close()

    # 3) 训练前后注意力对比（第0层第0头）
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, ttl in [(axes[0], attn_before, "训练前（随机初始化）"),
                         (axes[1], attn_after, "训练后")]:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(attn_tokens))); ax.set_yticks(range(len(attn_tokens)))
        ax.set_xticklabels(attn_tokens, fontsize=10); ax.set_yticklabels(attn_tokens, fontsize=10)
        ax.set_xlabel("被注意的字 (key)"); ax.set_ylabel("发起注意的字 (query)")
        ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("注意力权重：训练前 vs 训练后（L0 H0）"); plt.tight_layout()
    plt.savefig(fig_dir / "gpt_attn_before_after.png", dpi=140); plt.close()

    (ROOT / "ckpt").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "stoi": stoi, "itos": itos,
                "config": dict(vocab_size=len(chars), d_model=D_MODEL, n_heads=N_HEADS,
                               n_layers=N_LAYERS, max_len=BLOCK)}, ROOT / "ckpt" / "gpt.pt")
    (fig_dir / "gpt_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    print("\n" + "=" * 100)
    print("完成。图已存到 figures/：gpt_loss.png / gpt_param_evolution.png / gpt_attn_before_after.png")
    print(f"\n最终：train loss={history['train'][-1]:.3f}  val loss={history['val'][-1]:.3f}  "
          f"困惑度 PPL={math.exp(history['train'][-1]):.1f}（从 {len(chars)} 个字里平均锁定到 ~{math.exp(history['train'][-1]):.0f} 个）")


if __name__ == "__main__":
    main()
