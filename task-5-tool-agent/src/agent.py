"""手写 ReAct Agent:Thought/Action/Action Input/Observation 循环。

设计要点:
  - prompt 里列出工具 schema,约定模型每轮只输出一个 Action(JSON 参数)或 Final Answer
  - 用 stop=["Observation:"] 截断,防止模型自己幻觉出工具结果
  - 工具异常不中断循环:错误信息作为 Observation 喂回去,让模型自行纠错(错误恢复)
  - inject_error 钩子:实验用,把第 N 次工具结果替换成错误串,测恢复能力
LLM 后端:DashScope qwen-plus(OpenAI 兼容);README 建议的本地 Ollama 7B 同协议可切换。
"""
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from .tools import calculator, file_search, python_sandbox, wiki

REPO_ROOT = Path(__file__).resolve().parents[2]   # llm-beginner 仓库根(.env 在此)

TOOLS = {m.TOOL_SCHEMA["function"]["name"]: m
         for m in (calculator, python_sandbox, file_search, wiki)}

SYSTEM_TEMPLATE = """你是一个会使用工具的助手,用 ReAct 方式解决任务。

可用工具:
{tool_docs}

严格按以下格式输出,每轮只能有一个 Action:

Thought: 你对当前局面的思考
Action: 工具名(必须是上面列出的之一)
Action Input: 工具参数,必须是合法 JSON,如 {{"expression": "1+2"}}

输出 Action Input 后立即停止,等待 Observation(工具结果)。
得到足够信息后,输出:

Thought: 我已经得到答案
Final Answer: 最终答案(用中文,包含完整的关键数字/结论,不要省略)

规则:
- 数学计算用 calculator 或 python_sandbox,不要心算
- 需要文件信息用 file_search;需要百科事实用 wiki
- 工具报错时,分析错误原因,调整参数或换工具重试
- Final Answer 要直接回答任务问题,把关键数值写全"""

_ACTION_RE = re.compile(r"Action:\s*([\w\-]+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})\s*(?:$|Thought:|Action:)",
                       re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _load_key():
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY(环境变量或仓库根 .env)")
    return key


def _tool_docs():
    lines = []
    for name, mod in TOOLS.items():
        fn = mod.TOOL_SCHEMA["function"]
        params = json.dumps(fn["parameters"]["properties"], ensure_ascii=False)
        lines.append(f"- {name}: {fn['description']}\n  参数: {params}")
    return "\n".join(lines)


class ReActAgent:
    def __init__(self, model=None, max_steps=8, verbose=False,
                 inject_error_at=None, base_url=None, api_key=None):
        self.model = model or os.environ.get("AGENT_MODEL", "qwen-plus")
        self.max_steps = max_steps
        self.verbose = verbose
        self.inject_error_at = inject_error_at   # int:第 N 次(1-based)工具调用返回假错误
        self.client = OpenAI(
            api_key=api_key or _load_key(),
            base_url=base_url or os.environ.get(
                "AGENT_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

    def _chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0.2, max_tokens=1024, stop=["Observation:"],
        )
        return resp.choices[0].message.content or ""

    def run(self, task):
        """task -> AgentTrace: {"steps": [...], "final_answer": str, "success": bool}"""
        messages = [
            {"role": "system",
             "content": SYSTEM_TEMPLATE.format(tool_docs=_tool_docs())},
            {"role": "user", "content": f"任务:{task}"},
        ]
        steps = []
        n_tool_calls = 0

        for _ in range(self.max_steps):
            text = self._chat(messages)
            if self.verbose:
                print(f"\n--- LLM ---\n{text}")

            final = _FINAL_RE.search(text)
            if final:
                return {"steps": steps, "final_answer": final.group(1).strip(),
                        "success": True}

            action = _ACTION_RE.search(text)
            arg_m = _INPUT_RE.search(text)
            if not action:
                # 没按格式来:提醒一次,继续循环
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                                 "请严格按格式输出 Action + Action Input(JSON),"
                                 "或输出 Final Answer: 结束任务。"})
                continue

            tool_name = action.group(1).strip()
            try:
                tool_args = json.loads(arg_m.group(1)) if arg_m else {}
            except json.JSONDecodeError as e:
                tool_args, obs = {}, f"Action Input 不是合法 JSON: {e}"
            else:
                n_tool_calls += 1
                if self.inject_error_at and n_tool_calls == self.inject_error_at:
                    obs = "ERROR: tool crashed unexpectedly (injected failure)"
                elif tool_name not in TOOLS:
                    obs = f"未知工具 {tool_name!r},可用: {list(TOOLS)}"
                else:
                    try:
                        obs = str(TOOLS[tool_name].run(tool_args))
                    except Exception as e:
                        obs = f"工具执行出错: {type(e).__name__}: {e}"

            if self.verbose:
                print(f"--- Observation ---\n{obs[:400]}")
            steps.append({"thought": text.strip(), "tool": tool_name,
                          "tool_input": tool_args, "observation": obs[:2000]})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs[:2000]}"})

        return {"steps": steps, "final_answer": "", "success": False}
