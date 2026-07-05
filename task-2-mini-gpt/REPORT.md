# 任务二实验报告:从零实现 mini-GPT(BPE + RoPE + KV cache)

## 结果总览

| 自检项 | 结果 | 数值 |
|---|---|---|
| `tokenizer_roundtrip` | 通过 | 中英文样例 encode→decode 全部无损还原 |
| `kv_cache_equivalence` | 通过 | 增量 vs 全量 logits 最大误差 9.5e-06(< 1e-4) |
| `perplexity_on_dev` | 通过 | **dev PPL 24.1**(阈值 50) |

## 实现

- **BPE tokenizer**(`src/tokenizer.py`):手写 byte-level BPE。基础词表 = 256 字节,迭代 768 次"统计最高频相邻对 → 合并"得到 vocab 1024;byte-level 保证任意文本可逆往返。训练 35s(40 万字节采样),整体压缩率 2.16 字节/token(汉字 3 字节,即平均一个 token ≈ 0.7 字)
- **RoPE**(`src/rope.py`):维度两两配对旋转,角度 m·θᵢ;内积只依赖相对位置。增量解码时新 token 用 `offset=已缓存长度` 取 cos/sin——漏掉这个 offset 是 KV cache 自检最常见的挂法
- **Causal MHA + KV cache**(`src/attention.py`):缓存历史 K/V,每步只算新 token;T=1 时无需 causal mask(最后一个位置本就可见全部历史);T>1 时 mask 按全局位置 `triu(diagonal=past+1)`
- **模型**(`src/model.py`):Pre-LN decoder,6 层、d_model 384、6 heads,embedding 与 lm_head 权重绑定,共 **11.0M 参数**
- **采样**(`src/sampling.py`):greedy / temperature / top-k / top-p(核采样右移一位保证至少留一个 token)

## 训练

- 语料:全唐诗 7.6MB(232 万字,替换官方 48KB quick-start;README 允许换中文语料),90/10 切 train/dev,编码后 319 万 / 35.5 万 token
- 超参:block_size 256,batch 64,AdamW lr 3e-4(5% warmup + cosine),weight decay 0.1,clip 1.0,5000 步,RTX 4060 上 2.5 it/s 约 33 分钟
- 曲线(figures/training_curve.png):dev PPL 102(step 250)→ 49(step 1000)→ **29.2**(step 4750 最佳);step 3500 后 train/dev 开始分叉,按 dev 存最优起到 early stopping 作用

## 生成与采样策略对比(sample.py,prompt「白日依山盡，」)

- **greedy**:格律正确但陷入复读循环——"一片山中月""一樹花開盡"反复出现。确定性解码在语言模型上的典型退化
- **temp=0.8 + top-k/top-p**:多样性明显改善,五言结构、对仗意象基本成立("長寺無人意，松聲到客稀")
- **temp=1.2**:开始破碎,出现不通顺组合("興疑金見免")——温度过高时尾部低质量 token 被放大

## KV cache 速度实验(生成 256 token)

| 设备 | use_cache | 速度 |
|---|---|---|
| GPU (4060) | 开 / 关 | 194 vs 191 tok/s(**几乎无差**) |
| CPU | 开 / 关 | 220 vs 191 tok/s(+15%) |

诚实结论:在 11M 参数、≤512 上下文的设置下,瓶颈是每步的 kernel 启动/框架开销,重算前缀的矩阵乘对 GPU 近乎免费,所以 cache 收益很小;**cache 的收益随上下文长度与模型规模增长**(前缀重算是 O(N²) 的)。自检验证的是数值等价性,速度收益需要更大的设置才能体现。

## 其他观察

- **PPL 与分词粒度强相关**:同一模型,vocab 越大(token 越长),每 token 熵越高、PPL 越大;所以"PPL<50"只在给定 tokenizer 下有意义,跨 tokenizer 比较应换算成 bits-per-byte。本词表(1024)下 dev PPL 24.1 ≈ 每字节 2.12 bits
- 纯 Python BPE 编码 7.6MB 语料需约 70s,按行分块把贪心合并的复杂度限制在行内,缓存编码结果(data/*.ids.pt)避免重复开销

## 复现

```bash
python data/download.py        # 或换更大中文语料,保持 train/dev.txt 文件名
python train.py --steps 5000
python sample.py               # 采样策略 + KV cache 速度对比
python plot_training.py
python eval/run.py
```
