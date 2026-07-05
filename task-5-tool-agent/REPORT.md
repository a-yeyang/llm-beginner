# 任务五实验报告:手写 ReAct 工具调用 Agent

## 结果总览

| 自检项 | 结果 | 数值 |
|---|---|---|
| `tools_individual` | 通过 | 4/4 工具单测通过(wiki 联网正常,未跳过) |
| `multi_tool_success_rate` | 通过 | **10/10 = 100%**(通过线 60%) |
| `error_recovery`(可选项,自建实验) | 通过 | **9/10 = 90%**(通过线 40%) |

## 实现

- **4 个工具**(`src/tools/`),统一 `TOOL_SCHEMA`(OpenAI function calling 格式)+ `run(args)->str`:
  - `calculator`:**AST 白名单求值**(不用裸 eval):只放行数字/四则/幂/取余/math 函数,任何其他节点直接拒绝
  - `python_sandbox`:三层防护——正则静态黑名单(os/subprocess/open/exec 等)→ 子进程 `python -I` 隔离执行 → 10s 超时强杀;stdout 截断 2000 字符
  - `file_search`:文件名+内容双路匹配,跳过 .venv/models 等目录,命中文件附内容预览(所以"读 README 第一段"这类任务一次调用就够)
  - `wiki`:MediaWiki 搜索 API 选最相关标题 → REST 摘要 API;query 含中文走 zh、否则走 en
- **ReAct 循环**(`src/agent.py`,约 170 行):Thought/Action/Action Input/Observation 文本协议
  - `stop=["Observation:"]` 截断生成——**防止模型幻觉出工具结果**(不加这个,模型经常自己把 Observation 编出来)
  - 工具异常不中断循环,错误串作为 Observation 喂回,模型自行纠错
  - 格式解析失败时提醒一次并继续;`max_steps` 兜底
  - `inject_error_at` 钩子:第 N 次工具调用强制返回假错误,供错误恢复实验
- **LLM 后端**:DashScope qwen-plus(OpenAI 兼容);README 建议的本地 Ollama 同协议,改 `AGENT_BASE_URL` 即可切换

## 10 题任务集结果(eval/result.json)

全部 10 题答案关键词命中,含多工具组合题:
- 题 5(wiki 查 Hinton 出生年 → calculator 算年龄):agent 正确串联两个工具
- 题 8(sqrt(2026) 保留 6 位小数):agent 用 calculator 得原始值后,自行用 python_sandbox 做格式化输出
- 题 10(读文件第一段):file_search 的内容预览直接覆盖需求

## 错误恢复实验(run_error_recovery.py)

把每题**第 1 次工具调用**的 Observation 替换为 `"ERROR: tool crashed unexpectedly"`:
- 9/10 恢复成功;典型行为是模型在下一步 Thought 里说"工具好像出错了,我重试一次",然后带同样/修正过的参数重发 Action(平均比无错版本多 1-2 步)
- 失败的 1 题(题 3,统计 .md 文件数)在错误后转向了别的工具组合,答案格式偏离了关键词
- 说明:ReAct 的"错误也是 Observation"的设计天然具备重试能力,不需要显式的错误处理分支

## 观察与讨论

1. **stop 序列是文本 ReAct 最重要的工程细节**。没有它,强模型会一口气把 Thought/Action/Observation/Final Answer 全编完,工具形同虚设
2. 手写 ReAct(文本协议)vs 原生 function calling(结构化协议)的取舍:前者可解释、可移植到任何 completion 接口;后者参数解析零失败。本任务 10 题里 JSON 解析一次都没失败,qwen-plus 的格式服从性已足够好
3. 任务六用原生 function calling 实现 coding agent,正好与本任务形成两种协议的对照

## 复现

```bash
python data/download.py        # 生成 tasks.json + 文件夹具
python eval/run.py             # 三项自检(10 题约 3-5 分钟)
python run_error_recovery.py   # 错误恢复实验
```
