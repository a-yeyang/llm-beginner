# 任务一实验报告:从零实现 Transformer Encoder 做中文情感分类

## 结果总览

| 自检项 | 结果 | 数值 |
|---|---|---|
| `attention_correctness` | 通过 | 与 `F.scaled_dot_product_attention` 最大误差 7.2e-07(< 1e-5) |
| `causal_mask` | 通过 | 未来位置泄漏 0.0(< 1e-6) |
| `classifier_accuracy` | 通过 | **dev 准确率 0.8417**(通过线 0.80,参考基线 0.85) |

## 模型与训练配置

- 结构:字符级 embedding(词表 4252)+ 正弦位置编码 + 4 层 Pre-LN encoder block(d_model=128, 4 heads)+ masked mean pooling + 线性分类头,共 1.34M 参数
- 训练:AdamW(lr=3e-4, weight_decay=0.01),10% 线性 warmup + cosine 衰减,batch=32,6 epochs,梯度裁剪 1.0,按 dev acc 保存最优
- 数据:ChnSentiCorp(train 9600 / dev 1200),max_len=256
- 设备:CPU(torch 2.12+cpu),单次训练约几十分钟

## 训练过程

| epoch | train loss(均值) | dev acc |
|---|---|---|
| 1 | 0.598 | 0.7850 |
| 2 | 0.430 | 0.8100 |
| 3 | 0.387 | 0.8267 |
| 4 | 0.354 | **0.8417** |
| 5 | 0.336 | 0.8367 |
| 6 | 0.328 | 0.8383 |

第 4 个 epoch 后 train loss 仍在降但 dev acc 开始小幅回落,呈现轻微过拟合;"按 dev acc 存最优"起到了 early stopping 的作用。

## 实验观察(约 400 字)

最有意思的发现来自注意力热图的**层间对比**(figures/ 下 `attn_*_L0H0.png` 与 `attn_*_L3H0.png`,同一负面句子"送货太慢了,客服态度也很差,非常失望,再也不会买了"):

- **第 0 层**的注意力模式偏"结构性":副词"也""很"等 query 主要注意逗号等分隔符位置,整体分布较弥散,更像是在建立句子的局部/分段结构,还看不出与情感的直接关联。
- **第 3 层**出现了非常清晰的**垂直亮带**:几乎所有位置的 query 都把注意力集中到"太""差""不"这三个字上——恰好是这句话里全部的情感承载字("太慢""很差""不会买")。这说明经过多层堆叠,模型学会了把全句信息汇聚到判别性最强的 token 上,再经 mean pooling 送入分类头。这为"模型确实在看情感词"提供了直接证据,而不只是拟合表面统计。

两个工程细节对结果影响明显:一是 **padding mask** 必须以 `-inf` 填充打分而非乘 0,否则 softmax 后 PAD 位置仍分到概率;二是 pooling 阶段要做 **masked mean**(排除 PAD 位置),否则短句的句向量会被大量 PAD 位置稀释,早期实验里这会拖低约 1-2 个点的准确率。

另做了 causal mask 的 toy 语言模型预热(唐诗语料,见 `src/gpt.py` 与 `demo_gpt_train.py`),并延伸做了两组附加实验:过拟合对照(`gpt_anti_overfit.py`)与数据规模扫描(`gpt_data_scaling.py`),结果见 `figures/` 下的曲线与仪表盘,可作为任务二的预热。

## Definition of Done 勾选

- [x] M1 手写 `scaled_dot_product_attention`,自检通过(误差 7.2e-07)
- [x] M2 手写 `MultiHeadAttention` + `TransformerBlock`,前向形状正确
- [x] M3 ChnSentiCorp dev 准确率 0.8417 ≥ 0.80
- [x] M4 causal mask toy 语言模型,自检通过(泄漏 0.0)
- [x] M5 注意力热图 6 张(正面/负面/长句 × layer0/layer3)
- [ ] S1 head/层数消融(待做)
- [ ] S3 dev > 0.88(当前 0.8417)

## 复现

```bash
python data/download.py     # 注:datasets>=3.0 需改用 parquet 源(lansinuote/ChnSentiCorp)
python train.py --epochs 6
python visualize.py --layer 0 --head 0
python visualize.py --layer 3 --head 0
python eval/run.py
```
