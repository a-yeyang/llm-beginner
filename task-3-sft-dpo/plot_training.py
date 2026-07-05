"""Extract training logs and plot SFT + DPO curves."""
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent


def extract_sft_data(log_path: str):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    rows = []
    for m in re.finditer(
        r"Epoch (\d)/3.*?(\d+)/2500.*?loss=([\d.]+).*?lr=([\d.e+-]+)", raw
    ):
        epoch = int(m.group(1))
        step = int(m.group(2))
        loss = float(m.group(3))
        lr = float(m.group(4))
        global_step = (epoch - 1) * 2500 + step
        rows.append((global_step, loss, lr))
    return rows


def extract_dpo_data(log_path: str):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    rows = []
    for m in re.finditer(
        r"DPO Epoch (\d)/1.*?(\d+)/2499.*?loss=([\d.]+).*?margin=([\d.+-]+)",
        raw,
    ):
        step = int(m.group(2))
        loss = float(m.group(3))
        margin = float(m.group(4))
        rows.append((step, loss, margin))
    return rows


def smooth(values, weight=0.95):
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_sft(data, save_dir: Path):
    steps = [r[0] for r in data]
    losses = [r[1] for r in data]
    lrs = [r[2] for r in data]

    # Deduplicate: keep only unique steps (last value per step)
    seen = {}
    for s, l, lr in zip(steps, losses, lrs):
        seen[s] = (l, lr)
    steps_u = sorted(seen.keys())
    losses_u = [seen[s][0] for s in steps_u]
    lrs_u = [seen[s][1] for s in steps_u]

    # Subsample for plotting
    every = max(1, len(steps_u) // 500)
    steps_s = steps_u[::every]
    losses_s = losses_u[::every]
    lrs_s = lrs_u[::every]

    losses_smooth = smooth(losses_s, 0.95)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("SFT Training Curves (Qwen2.5-0.5B + LoRA r=8)",
                 fontsize=14, fontweight="bold")

    ax1.plot(steps_s, losses_s, alpha=0.2, color="steelblue", linewidth=0.5)
    ax1.plot(steps_s, losses_smooth, color="steelblue", linewidth=2,
             label="Loss (EMA 0.95)")

    for ep in [2500, 5000]:
        ax1.axvline(ep, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(1250, max(losses_smooth) * 0.98, "Epoch 1", ha="center",
             fontsize=10, color="gray")
    ax1.text(3750, max(losses_smooth) * 0.98, "Epoch 2", ha="center",
             fontsize=10, color="gray")
    ax1.text(6250, max(losses_smooth) * 0.98, "Epoch 3", ha="center",
             fontsize=10, color="gray")

    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_s, lrs_s, color="coral", linewidth=2)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))

    plt.tight_layout()
    out = save_dir / "sft_training_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_dpo(data, save_dir: Path):
    steps = [r[0] for r in data]
    losses = [r[1] for r in data]
    margins = [r[2] for r in data]

    seen_loss = {}
    seen_margin = {}
    for s, l, m in zip(steps, losses, margins):
        seen_loss[s] = l
        seen_margin[s] = m
    steps_u = sorted(seen_loss.keys())
    losses_u = [seen_loss[s] for s in steps_u]
    margins_u = [seen_margin[s] for s in steps_u]

    every = max(1, len(steps_u) // 500)
    steps_s = steps_u[::every]
    losses_s = losses_u[::every]
    margins_s = margins_u[::every]

    losses_smooth = smooth(losses_s, 0.95)
    margins_smooth = smooth(margins_s, 0.95)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("DPO Training Curves (beta=0.5, lr=1e-5, 1 epoch)",
                 fontsize=14, fontweight="bold")

    ax1.plot(steps_s, losses_s, alpha=0.2, color="steelblue", linewidth=0.5)
    ax1.plot(steps_s, losses_smooth, color="steelblue", linewidth=2,
             label="DPO Loss (EMA 0.95)")
    ax1.axhline(np.log(2), color="gray", linestyle=":", alpha=0.6,
                label=f"ln(2) ≈ {np.log(2):.3f} (random baseline)")
    ax1.set_ylabel("DPO Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_s, margins_s, alpha=0.2, color="seagreen", linewidth=0.5)
    ax2.plot(steps_s, margins_smooth, color="seagreen", linewidth=2,
             label="Reward Margin (EMA 0.95)")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_ylabel("Reward Margin", fontsize=12)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "dpo_training_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_combined_summary(sft_data, dpo_data, save_dir: Path):
    """Single overview figure with key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Task-3 训练总览：SFT + DPO on Qwen2.5-0.5B (RTX 4060)",
                 fontsize=14, fontweight="bold")

    # SFT Loss
    seen = {}
    for g, l, _ in sft_data:
        seen[g] = l
    steps = sorted(seen.keys())
    losses = [seen[s] for s in steps]
    every = max(1, len(steps) // 300)
    ax = axes[0]
    ax.plot(steps[::every], smooth([losses[i] for i in range(0, len(losses), every)], 0.95),
            color="steelblue", linewidth=2)
    ax.set_title("SFT Loss", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    for ep in [2500, 5000]:
        ax.axvline(ep, color="gray", linestyle="--", alpha=0.4)

    # DPO Loss
    seen = {}
    for s, l, m in dpo_data:
        seen[s] = (l, m)
    steps = sorted(seen.keys())
    every = max(1, len(steps) // 300)
    ax = axes[1]
    dl = smooth([seen[s][0] for s in steps[::every]], 0.95)
    ax.plot(steps[::every], dl, color="steelblue", linewidth=2)
    ax.axhline(np.log(2), color="gray", linestyle=":", alpha=0.6, label="ln(2)")
    ax.set_title("DPO Loss", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # DPO Reward Margin
    ax = axes[2]
    dm = smooth([seen[s][1] for s in steps[::every]], 0.95)
    ax.plot(steps[::every], dm, color="seagreen", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("DPO Reward Margin", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_log", type=str, required=True)
    parser.add_argument("--dpo_log", type=str, required=True)
    args = parser.parse_args()

    save_dir = ROOT / "figures"
    save_dir.mkdir(exist_ok=True)

    sft_data = extract_sft_data(args.sft_log)
    dpo_data = extract_dpo_data(args.dpo_log)
    print(f"SFT: {len(sft_data)} points, DPO: {len(dpo_data)} points")

    plot_sft(sft_data, save_dir)
    plot_dpo(dpo_data, save_dir)
    plot_combined_summary(sft_data, dpo_data, save_dir)


if __name__ == "__main__":
    main()
