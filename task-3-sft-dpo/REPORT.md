# 任务三实验报告:Qwen2.5-0.5B 的 SFT + DPO 两阶段对齐(手写 LoRA)

## 结果总览

| 自检项 | 结果 | 数值 |
|---|---|---|
| `lora_param_count` | 通过 | 可训练参数 540,672 / 494M,占比 **0.109%**(< 5%) |
| `loss_masking` | 通过 | assistant token 占比 0.571,落在 (0.2, 0.9) |
| `sft_vs_base` | 通过 | ckpt/sft 产出;质量对比见下文与 `eval/compare_output.txt` |

## 方法

- **基座**:Qwen2.5-0.5B(本地 `models/`),bfloat16,RTX 4060
- **手写 LoRA**(`src/lora.py`):`LoRALinear` 对冻结的 `W` 加低秩旁路 `x·A·B·(α/r)`,`inject_lora` 替换 `q_proj`/`v_proj`,r=8、α=16;训练时仅 LoRA 参数带梯度
- **chat template + loss masking**(`src/chat.py`):套 Qwen 的 `<|im_start|>role ... <|im_end|>` 模板;`build_labels` 把 system/user 段全部置 -100,只对 assistant 回复计 loss(实测 mask 后参与 loss 的 token 占 57.1%)
- **SFT**(`train_sft.py`):MOSS-003-SFT(no-tools)取 1 万条,3 epochs,lr 2e-4,batch 4 × grad_accum 4,max_len 512
- **DPO**(`train_dpo.py`):`hiyouga/DPO-En-Zh-20k`(zh)取 5 千对,从 SFT 权重出发,冻结副本作 reference model,β=0.1,2 epochs,lr 5e-5,batch 2 × grad_accum 4

## 训练曲线(figures/training_overview.png)

- **SFT loss**:约 1.16 → 1.10,三个 epoch 内平稳下降
- **DPO loss**:从 **ln(2)≈0.693 精确起步**——初始时 policy 与 reference 完全一致,implicit reward 差为 0,`-log σ(0) = ln 2`,这是 DPO 实现正确性的一个自带 sanity check;之后降至 ~0.66
- **DPO reward margin**(chosen 与 rejected 的 implicit reward 差):从 0 稳步升至 ~0.08,中途两次负向尖峰对应难样本 batch,整体趋势健康

## 三方生成对比(完整输出见 eval/compare_output.txt)

相同 3 个中文问题、相同采样参数(temperature 0.7, top-p 0.9)下:

| 模型 | 典型行为 |
|---|---|
| **base** | 能开头作答,但随即**崩坏**:泄漏 chat 模板标记、进入"自问自答"循环反复复读同一问题、夹杂无关多语 token(如 "wła""probante")——未对齐模型不知道"何时停" |
| **SFT** | 学会了指令格式:分点作答、结构规整、篇幅合理;但答完本题后偶发**跑题续写**(深度学习答完突然接了一段"性格类型") |
| **DPO** | 三题中回答最完整、切题,基本消除了跑题续写;主观可读性最好 |

**局限(如实记录)**:DPO 模型写诗时出现明显复读("五彩缤纷"连续 4 次、"生机勃勃"成串)。0.5B 基座 + 仅 q/v 的 rank-8 LoRA 的表达上限所致,推理端可用 repetition_penalty 缓解;这也说明 DPO 提升的是"偏好方向",并不能替基座补能力。

## 待做(实验建议中的可选项)

- [ ] LoRA rank 消融(4/8/16/32)
- [ ] 灾难性遗忘评估(C-Eval 子集 base vs SFT)
- [ ] 全量微调 vs LoRA 的显存/质量对照

## 复现

```bash
python data/download.py
python train_sft.py                 # ckpt/sft/lora_weights.pt
python train_dpo.py                 # ckpt/dpo/lora_weights.pt(从 SFT 出发)
python src/compare.py               # base/sft/dpo 三方生成对比
python plot_training.py             # figures/ 训练曲线
python eval/run.py                  # 三项自检
```
