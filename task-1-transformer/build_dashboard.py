"""把 demo_gpt_train.py 产出的 history + 图，组装成一个自包含的 HTML dashboard。

用法：python build_dashboard.py   （需先跑过 demo_gpt_train.py）
产物：figures/dashboard.html  —— 双击用浏览器打开即可，图片内嵌、离线可看。
"""
import base64
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def b64(path):
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    hist = json.loads((FIG / "gpt_history.json").read_text(encoding="utf-8"))
    meta = hist["meta"]
    steps = hist["step"]
    vocab = meta["vocab"]
    ln_vocab = math.log(vocab)
    val_min = min(hist["val"])
    val_min_step = steps[hist["val"].index(val_min)]

    # 相对偏移 = ||W-W0|| / ||W_init||（用 step0 的 wnorm 当 ||W_init||）
    rel_drift = {k: [d / (hist["wnorm"][k][0] + 1e-9) for d in hist["drift"][k]]
                 for k in hist["drift"]}

    samples_rows = "\n".join(
        f"<tr><td class='step'>{s}</td><td class='smp'>{t}</td></tr>"
        for s, t in zip(steps, hist["samples"]))

    data = {
        "steps": steps, "train": hist["train"], "val": hist["val"],
        "ppl": hist["ppl"], "grad": hist["grad_norm"], "drift": rel_drift,
    }

    html = TEMPLATE
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__ATTN__", b64(FIG / "gpt_attn_before_after.png"))
    html = html.replace("__LOSS_PNG__", b64(FIG / "gpt_loss.png"))
    html = html.replace("__PARAM_PNG__", b64(FIG / "gpt_param_evolution.png"))
    html = html.replace("__SAMPLES__", samples_rows)
    html = (html
            .replace("__NPARAMS__", f"{meta['n_params']/1e6:.2f}M")
            .replace("__VOCAB__", str(vocab))
            .replace("__DEVICE__", meta["device"].upper())
            .replace("__ARCH__", f"d_model={meta['d_model']} · {meta['n_layers']}层 · {meta['n_heads']}头")
            .replace("__STEP0__", f"{hist['train'][0]:.2f}")
            .replace("__LNVOCAB__", f"{ln_vocab:.2f}")
            .replace("__FINAL_TRAIN__", f"{hist['train'][-1]:.2f}")
            .replace("__FINAL_PPL__", f"{hist['ppl'][-1]:.1f}")
            .replace("__VALMIN__", f"{val_min:.2f}")
            .replace("__VALMINSTEP__", str(val_min_step))
            .replace("__NCHARS__", str(meta["n_chars"])))

    out = FIG / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"dashboard 已生成 -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>mini-GPT 训练 Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0f1419;--card:#1a2230;--ink:#e6edf3;--mut:#8b98a9;--acc:#4cc9f0;--warn:#f72585;--ok:#52b788}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--mut);margin:0 0 22px;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:26px}
.kpi{background:var(--card);border:1px solid #26303f;border-radius:12px;padding:14px 16px}
.kpi .v{font-size:22px;font-weight:700}
.kpi .l{color:var(--mut);font-size:12px;margin-top:3px}
.kpi .hint{color:var(--acc);font-size:11px;margin-top:5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--card);border:1px solid #26303f;border-radius:14px;padding:16px 16px 8px}
.card h3{margin:0 0 2px;font-size:15px}
.card p{margin:0 0 10px;color:var(--mut);font-size:12px}
.full{grid-column:1/3}
.chart{width:100%;height:320px}
img{width:100%;border-radius:8px;background:#fff}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #26303f;vertical-align:top}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
td.step{color:var(--acc);font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
td.smp{font-family:"Microsoft YaHei",serif;line-height:1.6;letter-spacing:.5px}
.scroll{max-height:430px;overflow:auto}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;margin-left:6px}
.tag.warn{background:rgba(247,37,133,.15);color:var(--warn)}
.tag.ok{background:rgba(82,183,136,.15);color:var(--ok)}
@media(max-width:820px){.grid{grid-template-columns:1fr}.full{grid-column:1}}
</style>
</head>
<body>
<div class="wrap">
  <h1>mini-GPT 训练 Dashboard <span class="tag ok">decoder-only</span></h1>
  <p class="sub">在唐诗语料(__NCHARS__ 字)上字符级训练 · 架构 __ARCH__ · 设备 __DEVICE__</p>

  <div class="kpis">
    <div class="kpi"><div class="v">__NPARAMS__</div><div class="l">参数量</div></div>
    <div class="kpi"><div class="v">__VOCAB__</div><div class="l">词表(不同字数)</div></div>
    <div class="kpi"><div class="v">__STEP0__</div><div class="l">step0 train loss</div><div class="hint">≈ ln(__VOCAB__)=__LNVOCAB__ 纯随机猜基线</div></div>
    <div class="kpi"><div class="v">__FINAL_TRAIN__</div><div class="l">最终 train loss</div><div class="hint">PPL __FINAL_PPL__（几乎背下来了）</div></div>
    <div class="kpi"><div class="v">__VALMIN__</div><div class="l">val loss 最低点</div><div class="hint">在 step __VALMINSTEP__ 触底后回升=过拟合</div></div>
  </div>

  <div class="grid">
    <div class="card"><h3>① Loss 曲线 <span class="tag warn">看 train↓ 但 val↑</span></h3>
      <p>train 一路降到接近 0（把训练集背下来了），val 在 step __VALMINSTEP__ 触底后回升——典型过拟合。</p>
      <div id="loss" class="chart"></div></div>

    <div class="card"><h3>② 困惑度 Perplexity</h3>
      <p>PPL=exp(loss)，直观含义「平均每个字从多少个里猜」。从 ~2540 个 → 个位数。</p>
      <div id="ppl" class="chart"></div></div>

    <div class="card"><h3>③ 各参数偏离随机初始的相对幅度</h3>
      <p>||W - W_init|| / ||W_init||。词嵌入动得最多（在学每个字的含义）；LayerNorm 的 γ 动得最少。</p>
      <div id="drift" class="chart"></div></div>

    <div class="card"><h3>④ 全局梯度范数（学习信号强度）</h3>
      <p>反向传播回来的梯度总长度。后期反而变大——模型在把记住的细节「钉得更死」。</p>
      <div id="grad" class="chart"></div></div>

    <div class="card full"><h3>⑤ 注意力：训练前 vs 训练后（第0层第0头）</h3>
      <p>两张都是下三角(causal mask 挡住了未来)。训练前权重在允许范围内近乎均匀；训练后明显聚焦到特定的前文字——注意力「学会了该看哪里」。</p>
      <img src="__ATTN__" alt="attention before/after"></div>

    <div class="card full"><h3>⑥ 生成样本随训练的演化</h3>
      <p>同一个起点反复采样。从随机乱码 → 带标点的诗形 → 像模像样的唐诗片段（其实是把训练集背出来了）。</p>
      <div class="scroll"><table><thead><tr><th>step</th><th>生成内容（temperature=0.8, top-k=20）</th></tr></thead>
        <tbody>__SAMPLES__</tbody></table></div></div>

    <details class="card full"><summary style="cursor:pointer;color:var(--mut)">静态图备份（无网络也能看）</summary>
      <img src="__LOSS_PNG__" style="margin-top:12px"><img src="__PARAM_PNG__" style="margin-top:12px"></details>
  </div>
</div>

<script>
const D = __DATA__;
const dark = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#8b98a9',size:12},margin:{l:48,r:16,t:10,b:40},
  xaxis:{gridcolor:'#26303f',title:'step',zeroline:false},
  yaxis:{gridcolor:'#26303f',zeroline:false},legend:{orientation:'h',y:1.15}};
const cfg={displayModeBar:false,responsive:true};

Plotly.newPlot('loss',[
  {x:D.steps,y:D.train,name:'train',mode:'lines+markers',line:{color:'#4cc9f0',width:2}},
  {x:D.steps,y:D.val,name:'val',mode:'lines+markers',line:{color:'#f72585',width:2}}
],{...dark,yaxis:{...dark.yaxis,title:'cross-entropy'}},cfg);

Plotly.newPlot('ppl',[
  {x:D.steps,y:D.ppl,name:'train PPL',mode:'lines+markers',line:{color:'#52b788',width:2}}
],{...dark,yaxis:{...dark.yaxis,title:'perplexity',type:'log'}},cfg);

Plotly.newPlot('grad',[
  {x:D.steps,y:D.grad,mode:'lines+markers',line:{color:'#ffba08',width:2}}
],{...dark,yaxis:{...dark.yaxis,title:'grad L2 norm'}},cfg);

const palette=['#4cc9f0','#f72585','#52b788','#ffba08'];
const dtraces=Object.keys(D.drift).map((k,i)=>(
  {x:D.steps,y:D.drift[k],name:k,mode:'lines+markers',line:{color:palette[i%4],width:2}}));
Plotly.newPlot('drift',dtraces,{...dark,yaxis:{...dark.yaxis,title:'相对偏移'}},cfg);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
