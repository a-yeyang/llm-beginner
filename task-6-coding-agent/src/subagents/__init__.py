"""Subagent:独立 context 的子 agent。

主 agent 把可隔离的子任务(代码搜索、测试诊断)派发给 subagent,
subagent 用自己的对话历史跑完整个工具循环,只把**摘要**交回主 agent——
这样主 agent 的 context 不被文件全文和冗长测试日志淹没。
"""
from .explorer import ExplorerSubagent
from .tester import TesterSubagent

SUBAGENTS = {
    "explorer": ExplorerSubagent,
    "tester": TesterSubagent,
}


def dispatch(agent_type, task, repo_path, llm_chat):
    """运行一个 subagent,返回其摘要文本。llm_chat: (messages, tools)->message。"""
    if agent_type not in SUBAGENTS:
        return f"错误:未知 subagent {agent_type!r},可用 {sorted(SUBAGENTS)}"
    return SUBAGENTS[agent_type](llm_chat).run(task, repo_path)
