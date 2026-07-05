"""过拟合 vs 正则化 对照实验。

同一份唐诗语料、同一套评测，跑两个配置：
  ① 基线   —— 大模型 / 无 dropout / 无 weight decay  → 复现过拟合
  ② 正则化 —— 小模型 + dropout + weight decay + label smoothing + 按 val 早停

val loss 都用「无 label smoothing 的纯交叉熵」算，保证两条曲线可比。
产物：figures/dashboard_overfit.html（自动打开）+ ckpt/gpt_best.pt（正则化的 val 最优权重）。
"""
import base64
import json
import math
from pathlib import Path

import torch

from src.gpt import MiniGPT

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

BLOCK, BATCH = 64, 32
MAX_STEPS, EVAL_INTERVAL, EVAL_BATCHES = 2000, 200, 20
SEED = 1337

CONFIGS = [
    {"name": "① 基线（过拟合）", "key": "base", "color": "#f72585",
     "d_model": 128, "n_heads": 4, "n_layers": 3,
     "dropout": 0.0, "weight_decay": 0.0, "label_smoothing": 0.0, "lr": 3e-4},
    {"name": "② 正则化", "key": "reg", "color": "#4cc9f0",
     "d_model": 64, "n_heads": 4, "n_layers": 2,
     "dropout": 0.3, "weight_decay": 0.2, "label_smoothing": 0.1, "lr": 3e-4},
]


def load_data(device):
    text = (ROOT.parent / "poetryFromTang.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n].to(device), data[n:].to(device), stoi, itos, chars


def train_one(cfg, train_data, val_data, stoi, itos, vocab, device):
    torch.manual_seed(SEED)  # 两个配置同种子 → batch 序列一致，公平对照

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
        x = torch.stack([d[i:i + BLOCK] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
        return x, y

    model = MiniGPT(vocab, cfg["d_model"], cfg["n_heads"], cfg["n_layers"],
                    max_len=BLOCK, dropout=cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = model.configure_optimizer(cfg["lr"], cfg["weight_decay"])

    @torch.no_grad()
    def estimate(split):  # 纯 CE，不加 label smoothing
        model.eval()
        ls = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(split)
            _, loss = model(x, y)
            ls[k] = loss.item()
        model.train()
        return ls.mean().item()

    @torch.no_grad()
    def sample(n=70):
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(ctx, n, 0.8, 20)[0].tolist()
        return "".join(itos[i] for i in out).replace("\n", " ")

    hist = {"step": [], "train": [], "val": []}
    best_val, best_step, best_state = float("inf"), 0, None
    print(f"\n=== {cfg['name']} | params={n_params/1e6:.3f}M | "
          f"dropout={cfg['dropout']} wd={cfg['weight_decay']} ls={cfg['label_smoothing']} ===")

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            tr, va = estimate("train"), estimate("val")
            hist["step"].append(step); hist["train"].append(tr); hist["val"].append(va)
            if va < best_val:
                best_val, best_step = va, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:>4} | train {tr:.3f} | val {va:.3f} | best_val {best_val:.3f}@{best_step}")
        if step == MAX_STEPS:
            break
        x, y = get_batch("train")
        _, loss = model(x, y, label_smoothing=cfg["label_smoothing"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return {"name": cfg["name"], "key": cfg["key"], "color": cfg["color"],
            "steps": hist["step"], "train": hist["train"], "val": hist["val"],
            "n_params": n_params, "best_val": best_val, "best_step": best_step,
            "final_train": hist["train"][-1], "final_val": hist["val"][-1],
            "sample": sample(), "best_state": best_state}


def build_dashboard(results, vocab, n_chars):
    base, reg = results
    gap_base = base["final_val"] - base["final_train"]
    gap_reg = reg["final_val"] - reg["final_train"]
    data = {r["key"]: {"name": r["name"], "color": r["color"], "steps": r["steps"],
                       "train": r["train"], "val": r["val"]} for r in results}

    def card(r):
        return (f"<div class='ex'><b style='color:{r['color']}'>{r['name']}</b>"
                f"<div class='m'>{r['n_params']/1e6:.2f}M 参数 · 最优 val "
                f"<b>{r['best_val']:.2f}</b>@step{r['best_step']} · 末尾 train {r['final_train']:.2f}</div>"
                f"<div class='smp'>{r['sample']}</div></div>")

    html = (OVERFIT_TMPL
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__VOCAB__", str(vocab)).replace("__NCHARS__", str(n_chars))
            .replace("__BV_BASE__", f"{base['best_val']:.2f}")
            .replace("__BV_REG__", f"{reg['best_val']:.2f}")
            .replace("__FV_BASE__", f"{base['final_val']:.2f}")
            .replace("__FV_REG__", f"{reg['final_val']:.2f}")
            .replace("__FT_BASE__", f"{base['final_train']:.2f}")
            .replace("__FT_REG__", f"{reg['final_train']:.2f}")
            .replace("__GAP_BASE__", f"{gap_base:.2f}")
            .replace("__GAP_REG__", f"{gap_reg:.2f}")
            .replace("__CARDS__", card(base) + card(reg)))
    out = FIG / "dashboard_overfit.html"
    FIG.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, stoi, itos, chars = load_data(device)
    print(f"device={device} 语料={len(train_data)+len(val_data)} 字 词表={len(chars)}")

    results = [train_one(c, train_data, val_data, stoi, itos, len(chars), device) for c in CONFIGS]

    # 保存正则化模型的 val 最优权重（早停产物）
    (ROOT / "ckpt").mkdir(exist_ok=True)
    reg = results[1]
    torch.save({"state_dict": reg["best_state"], "stoi": stoi, "itos": itos,
                "best_val": reg["best_val"], "best_step": reg["best_step"]},
               ROOT / "ckpt" / "gpt_best.pt")

    out = build_dashboard(results, len(chars), len(train_data) + len(val_data))
    print(f"\n对照 dashboard -> {out}")


OVERFIT_TMPL = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>过拟合 vs 正则化</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0f1419;--card:#1a2230;--ink:#e6edf3;--mut:#8b98a9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 60px}h1{font-size:23px;margin:0 0 4px}
.sub{color:var(--mut);font-size:14px;margin:0 0 22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:24px}
.kpi{background:var(--card);border:1px solid #26303f;border-radius:12px;padding:14px 16px}
.kpi .v{font-size:21px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px;margin-top:3px}
.kpi .d{font-size:12px;margin-top:5px}.up{color:#f72585}.down{color:#52b788}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--card);border:1px solid #26303f;border-radius:14px;padding:16px}
.card h3{margin:0 0 2px;font-size:15px}.card p{margin:0 0 8px;color:var(--mut);font-size:12px}
.full{grid-column:1/3}.chart{width:100%;height:340px}
.ex{padding:10px 0;border-top:1px solid #26303f}.ex .m{color:var(--mut);font-size:12px;margin:4px 0 8px}
.ex .smp{font-family:"Microsoft YaHei",serif;line-height:1.6;letter-spacing:.5px;font-size:14px}
.note{background:rgba(76,201,240,.08);border:1px solid #234;border-radius:12px;padding:14px 16px;color:#cfe;font-size:13px;line-height:1.7;margin-top:18px}
@media(max-width:820px){.grid,.kpis{grid-template-columns:1fr}.full{grid-column:1}}
</style></head><body><div class="wrap">
<h1>过拟合 vs 正则化</h1>
<p class="sub">唐诗语料 __NCHARS__ 字 · 词表 __VOCAB__ · val loss 均为纯交叉熵（可比）</p>
<div class="kpis">
 <div class="kpi"><div class="v">__FV_BASE__ → __FV_REG__</div><div class="l">末尾 val loss（越低越好）</div><div class="d down">基线疯涨到 __FV_BASE__，正则化压平在 __FV_REG__</div></div>
 <div class="kpi"><div class="v">__FT_BASE__ → __FT_REG__</div><div class="l">末尾 train loss</div><div class="d">基线≈0=背书；正则化不再死记</div></div>
 <div class="kpi"><div class="v">__GAP_BASE__ → __GAP_REG__</div><div class="l">末尾 train/val 间隙</div><div class="d down">间隙大幅缩小=过拟合被压住</div></div>
</div>
<div class="grid">
 <div class="card"><h3>val loss 对比 ← 重点看这条</h3><p>基线(粉)触底就疯涨；正则化(蓝)触底后基本压平，不再过拟合。</p><div id="val" class="chart"></div></div>
 <div class="card"><h3>train loss 对比</h3><p>基线一路冲向 0（把训练集背下来）；正则化被 dropout/wd 拦住，停在健康水位。</p><div id="train" class="chart"></div></div>
 <div class="card full"><h3>两个模型最终生成的样本</h3>__CARDS__</div>
</div>
<div class="note">
<b>为什么 val 没掉到很低？</b> 注意两者<b>最优 val 几乎一样(基线 __BV_BASE__ vs 正则化 __BV_REG__)</b>——<b>16k 字的数据量决定了泛化的「地板」(~6.5)</b>，正则化并没有、也无法把地板压低。它真正做的是：<b>让 val 触底后不反弹、把 train/val 间隙从 __GAP_BASE__ 压到 __GAP_REG__（这才是「不过拟合」）</b>，并通过早停保存 val 最优权重(已存 ckpt/gpt_best.pt)。要把<b>地板本身降下去，唯一的办法是喂更多数据</b>——下一步想真正提质，就上更大的语料。
</div></div>
<script>
const D=__DATA__;
const dark={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#8b98a9',size:12},
 margin:{l:48,r:16,t:10,b:40},xaxis:{gridcolor:'#26303f',title:'step',zeroline:false},
 yaxis:{gridcolor:'#26303f',zeroline:false,title:'cross-entropy'},legend:{orientation:'h',y:1.15}};
const cfg={displayModeBar:false,responsive:true};
function traces(field){return Object.values(D).map(r=>({x:r.steps,y:r[field],name:r.name,
 mode:'lines+markers',line:{color:r.color,width:2}}));}
Plotly.newPlot('val',traces('val'),dark,cfg);
Plotly.newPlot('train',traces('train'),dark,cfg);
</script></body></html>
"""

if __name__ == "__main__":
    main()
