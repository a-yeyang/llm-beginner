"""Subagent 基类:带工具子集的小型 agent loop,产出摘要。"""
import json

from ..mcp_server import call_tool, list_tools


class BaseSubagent:
    """子类定义 NAME / SYSTEM / TOOL_NAMES / MAX_STEPS。"""

    NAME = "base"
    SYSTEM = ""
    TOOL_NAMES = ()
    MAX_STEPS = 6

    def __init__(self, llm_chat):
        self.llm_chat = llm_chat          # 复用主 agent 的模型调用(独立 messages)

    def _tools(self):
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["inputSchema"]}}
                for t in list_tools() if t["name"] in self.TOOL_NAMES]

    def run(self, task, repo_path):
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user",
             "content": f"仓库路径:{repo_path}\n子任务:{task}\n"
                        f"完成后直接输出最终摘要(不要再调用工具)。"},
        ]
        for _ in range(self.MAX_STEPS):
            msg = self.llm_chat(messages, self._tools())
            if not msg.tool_calls:
                return msg.content or "(subagent 无输出)"
            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = call_tool(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": obs[:4000]})
        return "(subagent 达到步数上限,未产出摘要)"
