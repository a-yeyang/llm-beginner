"""画 task-2 训练曲线:train/dev loss 与 dev 困惑度。"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
h = json.loads((ROOT / "figures" / "history.json").read_text(encoding="utf-8"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(h["step"], h["train_loss"], label="train loss")
ax1.plot(h["step"], h["dev_loss"], label="dev loss")
ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_title("mini-GPT loss (poetry, 11M params)")

ppl = [math.exp(x) for x in h["dev_loss"]]
ax2.plot(h["step"], ppl, color="tab:green")
ax2.axhline(50, ls="--", color="gray", label="pass threshold 50")
ax2.set_xlabel("step"); ax2.set_ylabel("dev perplexity"); ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_title("dev perplexity")

fig.tight_layout()
out = ROOT / "figures" / "training_curve.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
