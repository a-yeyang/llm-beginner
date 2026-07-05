# 任务六实验报告:Mini Coding Agent(MCP + Skill + Subagent 三层栈)

## 结果总览

| 自检项 | 结果 | 说明 |
|---|---|---|
| `mcp_server_lists_tools` | 通过 | 7 个工具(要求 ≥5) |
| `skill_loader_metadata` | 通过 | 3 个 Skill,name/description 齐全 |
| `toy_repo_patch` | 通过 | agent 自主修复 `calculator.add`,pytest 3 passed,trace 含 patch |
| `swebench_lite_sample` | 跳过 | 可选进阶,需 `--with-swebench` + 本地 clone 对应 repo(未来工作) |

## 三层栈实现

**底层 Tools / MCP**(`src/mcp_server.py`):手写 JSON-RPC 2.0 over stdio 的极简 MCP server(不依赖 SDK),响应 `initialize` / `tools/list` / `tools/call`,可被任意 MCP 客户端接入;同一套工具注册表同时暴露进程内接口(`list_tools`/`call_tool`),agent 走进程内路径免掉 IPC。7 个无状态工具:`read_file`(带行号)/ `write_file`(整文件覆盖)/ `list_files` / `search_code`(正则)/ `run_tests`(pytest 子进程)/ `git_diff` / `git_apply`。

**中层 Skills**(`src/skill_loader.py` + `src/skills/`):SKILL.md = frontmatter(name/description)+ workflow 正文。渐进式披露:agent 平时只见目录(便宜),`match()` 按关键词重叠命中后才 `load()` 全文注入 system prompt。三个 skill:`test-runner`(测试驱动修复流程)、`code-review`、`pr-description-writer`。

**顶层 Subagents**(`src/subagents/`):`explorer`(代码搜索,只读工具)与 `tester`(测试诊断)在**独立 messages** 里跑自己的工具循环,只把摘要交回主 agent——context 隔离,文件全文和测试日志不进主 agent 历史。

**主循环**(`src/agent.py`):OpenAI 原生 function calling(与任务五的手写文本 ReAct 形成协议对照),`while not done: model → tool → observation`;`write_file` 前自动快照原文件,结束时用 difflib 产出 unified diff patch(不依赖 git 状态);结束条件 = 模型输出 FINISHED 或步数上限,最后统一跑 pytest 定 `tests_passed`。

## toy-repo 修复轨迹(典型一次)

1. `run_tests` → 发现 `test_add_positive_numbers` 失败(`assert add(2,3)==5` got -1)
2. `read_file calculator.py` → 定位 `return a - b`
3. `write_file` → 改为 `return a + b`
4. `run_tests` → `3 passed`,输出 FINISHED
patch:`- return a - b` / `+ return a + b`

## 消融实验(eval/ablation.json,每配置 2 次)

| 配置 | 成功率 | 工具步数 |
|---|---|---|
| baseline(纯 prompt) | 2/2 | 6, 6 |
| **+Skill(test-runner)** | 2/2 | **4, 4** |
| +Subagent | 2/2 | 6, 5 |
| +Skill+Subagent | 2/2 | 5, 5 |

**观察**:
1. **Skill 稳定减少 1/3 步数**:SKILL.md 里"先跑测试、拿断言反推实现错误"的流程让 agent 跳过盲目探索,直奔 run_tests → read → fix。这正是 Skill 的定位——把工程经验做成可复用能力包,比堆长 system prompt 干净
2. **Subagent 在小任务上不赚**:toy-repo 一共 4 个文件,派发 explorer 的往返开销抵消了 context 隔离收益。Subagent 的价值要在大仓库(文件多到淹没主 context)才体现——这是诚实的负结果
3. qwen-plus 的 function calling 参数解析 8 次运行零失败;与任务五文本 ReAct 相比,结构化协议在"工具参数复杂"(如 write_file 的长 content)场景明显更稳

## 局限与未来工作

- SWE-bench Lite 未跑:需 clone 对应版本仓库 + 复杂环境隔离,已留好接口(自检第 4 项),是下一步的主要挑战
- write_file 是整文件覆盖,大文件场景应换成基于行号/锚点的局部编辑工具
- Skill 匹配是朴素关键词重叠,更严谨可用 embedding 相似度

## 复现

```bash
python data/download.py          # 生成 toy-repo
python src/mcp_server.py         # (可选)独立启动 MCP server 验证 stdio 协议
python eval/run.py               # 四项自检
python run_ablation.py           # Skill/Subagent 消融
```
