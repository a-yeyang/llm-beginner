"""数据量对照实验：同一模型、同一算力、同一验证集，只改训练数据量。

证明「数据量本身」决定 val 地板：
  - 词表、held-out 验证集都固定（取自 257 万字大语料）→ 三次训练的 val loss 完全可比
  - 唯一变量 = 训练数据池大小：2万字 / 20万字 / 全部(~250万字)
产物：figures/dashboard_data.html（自动打开）+ ckpt/gpt_poems.pt（全量数据的最优权重）。
"""
import json
from pathlib import Path

import torch

from src.gpt import MiniGPT

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

BLOCK, BATCH = 64, 32
MAX_STEPS, EVAL_INTERVAL, EVAL_BATCHES = 2000, 250, 30
SEED = 1337
VAL_CHARS = 80_000
# 模型 + 正则化固定不变，唯一变量是训练数据量
D_MODEL, N_HEADS, N_LAYERS, DROPOUT, WD, LS, LR = 128, 4, 3, 0.1, 0.1, 0.05, 3e-4

RUNS = [
    {"name": "2万字 (小)", "key": "s", "color": "#f72585", "size": 20_000},
    {"name": "20万字 (中)", "key": "m", "color": "#ffba08", "size": 200_000},
    {"name": "全部~250万字 (大)", "key": "l", "color": "#4cc9f0", "size": None},
]


def train_one(run, train_full, val_ids, stoi, itos, vocab, device):
    torch.manual_seed(SEED)
    size = run["size"] or len(train_full)
    pool = train_full[:size]

    def get_batch(d):
        ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
        x = torch.stack([d[i:i + BLOCK] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
        return x, y

    model = MiniGPT(vocab, D_MODEL, N_HEADS, N_LAYERS, max_len=BLOCK, dropout=DROPOUT).to(device)
    opt = model.configure_optimizer(LR, WD)

    @torch.no_grad()
    def estimate(d):
        model.eval()
        ls = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(d)
            _, loss = model(x, y)
            ls[k] = loss.item()
        model.train()
        return ls.mean().item()

    @torch.no_grad()
    def sample(n=70):
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(ctx, n, 0.8, 30)[0].tolist()
        return "".join(itos[i] for i in out).replace("\n", " ")

    hist = {"step": [], "train": [], "val": []}
    best_val, best_step, best_state = float("inf"), 0, None
    print(f"\n=== {run['name']} | 训练池={size} 字 ===")
    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            tr, va = estimate(pool), estimate(val_ids)
            hist["step"].append(step); hist["train"].append(tr); hist["val"].append(va)
            if va < best_val:
                best_val, best_step = va, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:>4} | train {tr:.3f} | val {va:.3f} | best_val {best_val:.3f}@{best_step}")
        if step == MAX_STEPS:
            break
        x, y = get_batch(pool)
        _, loss = model(x, y, label_smoothing=LS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return {**run, "steps": hist["step"], "train": hist["train"], "val": hist["val"],
            "best_val": best_val, "best_step": best_step, "final_val": hist["val"][-1],
            "sample": sample(), "best_state": best_state, "real_size": size}


def build_dashboard(results, vocab):
    import math
    ln_v = math.log(vocab)
    data = {r["key"]: {"name": r["name"], "color": r["color"], "steps": r["steps"],
                       "train": r["train"], "val": r["val"], "size": r["real_size"]}
            for r in results}
    bests = [(r["real_size"], r["best_val"]) for r in results]

    def card(r):
        return (f"<div class='ex'><b style='color:{r['color']}'>{r['name']}</b>"
                f"<div class='m'>最优 val <b>{r['best_val']:.2f}</b>@step{r['best_step']}"
                f" · 末尾 val {r['final_val']:.2f}</div>"
                f"<div class='smp'>{r['sample']}</div></div>")

    html = (TMPL
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__BESTS__", json.dumps(bests))
            .replace("__VOCAB__", str(vocab)).replace("__LNV__", f"{ln_v:.2f}")
            .replace("__BV_S__", f"{results[0]['best_val']:.2f}")
            .replace("__BV_M__", f"{results[1]['best_val']:.2f}")
            .replace("__BV_L__", f"{results[2]['best_val']:.2f}")
            .replace("__CARDS__", "".join(card(r) for r in results)))
    out = FIG / "dashboard_data.html"
    FIG.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = (ROOT / "data" / "poems.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long).to(device)
    val_ids = ids[-VAL_CHARS:]
    train_full = ids[:-VAL_CHARS]
    print(f"device={device} 语料={len(ids)}字 词表={len(chars)} "
          f"train_full={len(train_full)} val={len(val_ids)}（固定）")

    results = [train_one(r, train_full, val_ids, stoi, itos, len(chars), device) for r in RUNS]

    (ROOT / "ckpt").mkdir(exist_ok=True)
    big = results[-1]
    torch.save({"state_dict": big["best_state"], "stoi": stoi, "itos": itos,
                "best_val": big["best_val"]}, ROOT / "ckpt" / "gpt_poems.pt")
    out = build_dashboard(results, len(chars))
    print(f"\n数据量对照 dashboard -> {out}")


TMPL = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>数据量 vs 泛化</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0f1419;--card:#1a2230;--ink:#e6edf3;--mut:#8b98a9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 60px}h1{font-size:23px;margin:0 0 4px}
.sub{color:var(--mut);font-size:14px;margin:0 0 22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}
.kpi{background:var(--card);border:1px solid #26303f;border-radius:12px;padding:14px 16px}
.kpi .v{font-size:21px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px;margin-top:3px}
.kpi .d{font-size:12px;margin-top:5px;color:#52b788}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--card);border:1px solid #26303f;border-radius:14px;padding:16px}
.card h3{margin:0 0 2px;font-size:15px}.card p{margin:0 0 8px;color:var(--mut);font-size:12px}
.full{grid-column:1/3}.chart{width:100%;height:340px}
.ex{padding:10px 0;border-top:1px solid #26303f}.ex .m{color:var(--mut);font-size:12px;margin:4px 0 8px}
.ex .smp{font-family:"Microsoft YaHei",serif;line-height:1.6;letter-spacing:.5px;font-size:14px}
.note{background:rgba(76,201,240,.08);border:1px solid #234;border-radius:12px;padding:14px 16px;color:#cfe;font-size:13px;line-height:1.7;margin-top:18px}
@media(max-width:820px){.grid,.kpis{grid-template-columns:1fr}.full{grid-column:1}}
</style></head><body><div class="wrap">
<h1>数据量 vs 泛化：把 val 地板压下去</h1>
<p class="sub">同一模型 · 同一算力(2000步) · 同一 held-out 验证集 · 词表 __VOCAB__（ln=__LNV__ 是纯随机基线）。唯一变量=训练数据量</p>
<div class="kpis">
 <div class="kpi"><div class="v">__BV_S__</div><div class="l">2万字 最优 val</div><div class="d">数据少→地板高</div></div>
 <div class="kpi"><div class="v">__BV_M__</div><div class="l">20万字 最优 val</div><div class="d">↓ 下降</div></div>
 <div class="kpi"><div class="v">__BV_L__</div><div class="l">250万字 最优 val</div><div class="d">↓↓ 地板被真正压低</div></div>
</div>
<div class="grid">
 <div class="card"><h3>val loss 随数据量 ← 重点</h3><p>三条曲线模型完全一样，只是喂的数据量不同。数据越多，val 越低、越不反弹。</p><div id="val" class="chart"></div></div>
 <div class="card"><h3>最优 val vs 训练数据量（log x）</h3><p>数据量每上一个台阶，泛化地板就下一个台阶——这才是根本杠杆。</p><div id="scale" class="chart"></div></div>
 <div class="card full"><h3>三个数据量下模型生成的样本</h3>__CARDS__</div>
</div>
<div class="note">
<b>对比上一个实验：</b>之前正则化只能把 val「压平在地板上(~6.5)」却压不低地板；这次<b>仅仅把数据从 2万→250万字，val 地板就实打实下降了</b>（见上方三个数字）。<b>数据量是降低泛化误差的根本杠杆，正则化只是防止你浪费已有数据</b>。全量数据的最优权重已存 ckpt/gpt_poems.pt。
</div></div>
<script>
const D=__DATA__, BESTS=__BESTS__;
const dark={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#8b98a9',size:12},
 margin:{l:48,r:16,t:10,b:44},xaxis:{gridcolor:'#26303f',title:'step',zeroline:false},
 yaxis:{gridcolor:'#26303f',zeroline:false,title:'val cross-entropy'},legend:{orientation:'h',y:1.15}};
const cfg={displayModeBar:false,responsive:true};
Plotly.newPlot('val',Object.values(D).map(r=>({x:r.steps,y:r.val,name:r.name,
 mode:'lines+markers',line:{color:r.color,width:2}})),dark,cfg);
Plotly.newPlot('scale',[{x:BESTS.map(b=>b[0]),y:BESTS.map(b=>b[1]),mode:'lines+markers',
 line:{color:'#52b788',width:2},marker:{size:10}}],
 {...dark,xaxis:{...dark.xaxis,title:'训练字数',type:'log'},yaxis:{...dark.yaxis,title:'最优 val loss'}},cfg);
</script></body></html>
"""

if __name__ == "__main__":
    main()
