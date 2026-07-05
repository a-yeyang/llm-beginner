"""Python 沙箱工具:子进程隔离执行 + 静态黑名单 + 超时 + stdout 捕获。

隔离手段:
  1. 静态检查:拒绝 os/sys/subprocess/socket 等危险 import 与 open/exec/eval 调用
  2. 子进程 `python -I`(isolated mode,不继承 site/环境注入),崩溃不影响主进程
  3. 10s 超时强杀,防死循环
"""
import re
import subprocess
import sys

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限沙箱中执行一段 Python 代码并返回 stdout。"
            "必须用 print() 输出结果。可用 math 等标准库纯计算模块;"
            "禁止文件/网络/系统操作(os、open、subprocess 等)。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
            },
            "required": ["code"],
        },
    },
}

_BLOCKED = re.compile(
    r"\b(import\s+(os|sys|subprocess|socket|shutil|ctypes|pathlib|importlib)\b"
    r"|from\s+(os|sys|subprocess|socket|shutil|ctypes|pathlib|importlib)\b"
    r"|__import__|open\s*\(|exec\s*\(|eval\s*\(|input\s*\()"
)

TIMEOUT = 10


def run(args):
    code = str(args.get("code", ""))
    if not code.strip():
        return "错误:code 为空"
    m = _BLOCKED.search(code)
    if m:
        return f"错误:代码包含受限操作 {m.group(0)!r},沙箱拒绝执行"
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"错误:执行超过 {TIMEOUT}s 被终止(可能有死循环)"
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        return f"执行报错(exit {r.returncode}): {err[-1] if err else '未知错误'}"
    if not out:
        return "执行成功但没有输出;请在代码里用 print() 打印结果"
    return out[:2000]
