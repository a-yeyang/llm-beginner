"""Mini Coding Agent 主循环:while not done: model -> tool -> observation。

三层栈的集成方式:
  - Tools:进程内直连 mcp_server 的注册表(list_tools/call_tool),
    独立部署时同一套工具经 stdio MCP 协议暴露
  - Skills:SkillLoader 按 issue 关键词匹配,把命中的 SKILL.md 全文注入 system prompt
  - Subagents:暴露 dispatch_subagent 工具,explorer(代码搜索)/tester(测试诊断)
    在独立 context 里跑,只回传摘要

与任务五的对照:这里用 OpenAI 原生 function calling(结构化协议),
任务五是手写文本 ReAct——两种工具调用协议各自的工程取舍见 REPORT.md。
"""
import difflib
import json
import os
from pathlib import Path

from openai import OpenAI

from .mcp_server import call_tool, list_tools
from .skill_loader import SkillLoader

REPO_ROOT = Path(__file__).resolve().parents[2]   # llm-beginner 仓库根(.env)
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SYSTEM_TEMPLATE = """你是一个 mini coding agent,在本地仓库上修复 issue。
仓库路径:{repo_path}

工作流程:
1. 用 list_files / read_file / search_code 理解仓库(也可派 explorer subagent 并只看摘要)
2. 定位问题,用 write_file 修复(写完整文件内容;路径用 仓库路径/文件名)
3. 用 run_tests 验证(也可派 tester subagent 诊断失败)
4. 测试全部通过(exit_code=0)后,不再调用工具,输出一段中文总结,以 FINISHED 结尾

规则:
- 不要修改测试文件
- 做最小修改,不要顺手重构
- 测试不过就继续迭代,不要在未通过时输出 FINISHED
{skill_section}"""


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


class CodingAgent:
    def __init__(self, model=None, max_steps=16, use_skills=True,
                 use_subagents=True, verbose=False):
        self.model = model or os.environ.get("AGENT_MODEL", "qwen-plus")
        self.max_steps = max_steps
        self.use_skills = use_skills
        self.use_subagents = use_subagents
        self.verbose = verbose
        self.skill_loader = SkillLoader(SKILLS_DIR)
        self.client = OpenAI(
            api_key=_load_key(),
            base_url=os.environ.get(
                "AGENT_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

    # ----------------------------- LLM / 工具 -----------------------------

    def _chat(self, messages, tools):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools,
            temperature=0.1, max_tokens=2048,
        )
        return resp.choices[0].message

    def _tool_schemas(self):
        tools = [{"type": "function",
                  "function": {"name": t["name"], "description": t["description"],
                               "parameters": t["inputSchema"]}}
                 for t in list_tools()]
        if self.use_subagents:
            tools.append({"type": "function", "function": {
                "name": "dispatch_subagent",
                "description": ("派发子任务给独立 context 的 subagent,只返回摘要。"
                                "explorer=代码搜索定位;tester=跑测试并诊断失败"),
                "parameters": {"type": "object", "properties": {
                    "agent": {"type": "string", "enum": ["explorer", "tester"]},
                    "task": {"type": "string", "description": "子任务描述"},
                }, "required": ["agent", "task"]},
            }})
        return tools

    # ----------------------------- 补丁追踪 -----------------------------

    def _snapshot_before_write(self, args):
        path = Path(str(args.get("path", "")))
        if path not in self._originals:
            self._originals[path] = (
                path.read_text(encoding="utf-8", errors="replace")
                if path.exists() else "")

    def _make_patch(self, repo_path):
        diffs = []
        for path, before in self._originals.items():
            after = (path.read_text(encoding="utf-8", errors="replace")
                     if path.exists() else "")
            if before == after:
                continue
            try:
                rel = path.relative_to(repo_path)
            except ValueError:
                rel = path
            diffs.append("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}")))
        return "\n".join(diffs)

    # ----------------------------- 主循环 -----------------------------

    def run(self, repo_path, issue):
        """返回 Trace: {"steps": [...], "patch": str, "tests_passed": bool}。"""
        repo_path = str(Path(repo_path).resolve())
        self._originals = {}

        skill_section = ""
        if self.use_skills:
            hit = self.skill_loader.match(issue)
            if hit:
                skill_section = (f"\n已为本任务加载 Skill「{hit}」,遵循其流程:\n"
                                 f"{self.skill_loader.load(hit)}")

        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(
                repo_path=repo_path, skill_section=skill_section)},
            {"role": "user", "content": f"Issue:\n{issue}"},
        ]
        tools = self._tool_schemas()
        steps = []

        for _ in range(self.max_steps):
            msg = self._chat(messages, tools)
            if not msg.tool_calls:
                content = msg.content or ""
                steps.append({"type": "message", "content": content[:500]})
                if "FINISHED" in content:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                                 "若已完成请输出 FINISHED;否则继续调用工具。"})
                continue

            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "write_file":
                    self._snapshot_before_write(args)
                if name == "dispatch_subagent":
                    from .subagents import dispatch
                    obs = dispatch(args.get("agent", ""), args.get("task", ""),
                                   repo_path, self._chat)
                else:
                    obs = call_tool(name, args)
                if self.verbose:
                    print(f"[{name}] {json.dumps(args, ensure_ascii=False)[:120]}"
                          f"\n  -> {obs[:200]}")
                steps.append({"type": "tool", "tool": name,
                              "tool_input": args, "observation": obs[:1500]})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": obs[:4000]})

        final_test = call_tool("run_tests", {"dir": repo_path})
        tests_passed = final_test.startswith("exit_code=0")
        return {"steps": steps, "patch": self._make_patch(repo_path),
                "tests_passed": tests_passed}
