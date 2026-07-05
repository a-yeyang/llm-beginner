# 任务四实验报告:手写中文 RAG 流水线(BGE + FAISS + Reranker)

## 结果总览

| 自检项 | 结果 | 数值 |
|---|---|---|
| `chunking_sanity` | 通过 | 平均块长 248.2(要求 128–307) |
| `nndl_gold_recall_at_10` | 通过 | **Recall@10 = 0.833**(通过线 0.6),Recall@1 = 0.6,MRR = 0.656 |
| `rag_end_to_end` | 通过 | 返回带 [编号] 来源引用的答案 |

## 系统构成(全部手写,未用 LangChain/LlamaIndex)

```
kb.pdf(NNDL v2 教材,563 页/60.7 万字)
  → pypdf 逐页提取(记录页偏移供溯源)
  → chunker:句子边界感知打包,chunk 256 字 / overlap 64 → 3506 chunks
  → BGE-small-zh-v1.5 embedding(CLS pooling + L2 归一化,query 侧加检索指令前缀)
  → FAISS IndexFlatIP(归一化向量的内积 = 余弦相似度,精确检索)
  → 粗排 top-30 → bge-reranker-base(cross-encoder)精排 → top-k
  → qwen-plus 生成(DashScope API;prompt 要求仅依据资料作答+标注来源+不知则拒答)
```

两个实现说明:
- **不用 sentence-transformers**:该包在本机与 torch/pyarrow 组合加载时发生 DLL 冲突段错误,且它只是薄封装——直接用 transformers 加载 BGE(CLS pooling + normalize)等价且依赖更干净
- **生成端用 API 而非本地 7B**:8GB 显存装不下 bf16 的 Qwen2.5-7B;生成端在本系统里是可插拔组件,核心的检索侧全部本地

## 消融实验(30 题 gold QA,figures/ablation.png)

| chunk size | reranker | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| 128 | ✗ | 0.367 | 0.667 | 0.767 | 0.513 |
| 128 | ✓ | 0.533 | 0.733 | 0.833 | 0.633 |
| 256 | ✗ | 0.300 | 0.733 | 0.733 | 0.444 |
| 256 | ✓ | **0.600** | 0.800 | 0.833 | 0.656 |
| 512 | ✗ | 0.333 | 0.633 | 0.800 | 0.478 |
| 512 | ✓ | 0.467 | 0.833 | 0.867 | 0.614 |
| 1024 | ✗ | 0.400 | 0.767 | 0.867 | 0.553 |
| 1024 | ✓ | 0.633 | 0.867 | **0.900** | **0.749** |

**观察**:
1. **reranker 是提升最大的单一组件**:所有 chunk size 下 Recall@1 提升 0.17–0.30、MRR 提升 0.12–0.21。bi-encoder 粗排负责"把相关的捞进前 30",cross-encoder 负责"把最相关的排到最前"——两者分工正是两阶段检索的教科书动机
2. **大 chunk 召回指标更高但有陷阱**:1024 字块的 anchor 命中率天然占优(块越大越容易包含 anchor 子串),但 BGE 输入截断在 512 token,块的后半段根本没被编码——命中靠的是前半段语义;且大块喂给生成器噪声更多。综合选 **256 + rerank** 作为默认配置
3. 未命中的 5 题(如 FlashAttention/GQA、DQN/PPO 对比)多为"跨小节汇总型"问题,单块检索天然吃亏,适合 query 分解或多跳检索(未来工作)

## 端到端示例(自检第 1 题)

> **Q**: 前馈神经网络的基本特点是什么?
> **A**: 前馈神经网络的基本特点是:信息沿着网络从输入到输出单向传播,不存在反馈连接 [1];以神经元为基本单元,通过多层仿射变换与非线性变换的复合实现逐层映射 [3];……

过程中独立发现了官方 `eval/run.py` 的一个 bug:`ok = ... and r.get("sources")` 会让 `pass` 字段变成列表而非布尔值,导致评测壳写 result.json 时崩溃(unhashable type)。本地先以 `bool()` 修复;后发现官方在 `446a1ce` 提交中已做了等价(且更完整)的修复,变基时已采用官方版本。

## 复现

```bash
python data/download.py                # BGE 模型 + kb.pdf + gold QA 校验
python -c "from src.indexer import build_index; build_index()"
python eval/run.py                     # 三项自检
python run_experiments.py              # chunk × rerank 消融
```
