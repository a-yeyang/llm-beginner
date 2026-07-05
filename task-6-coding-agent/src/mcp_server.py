"""极简 MCP server:手写 JSON-RPC 2.0 over stdio + 工具注册表。

两种用法:
  1. `python src/mcp_server.py` 独立启动,按 MCP stdio 协议(newline-delimited JSON-RPC)
     响应 initialize / tools/list / tools/call,可被任意 MCP 客户端接入
  2. 进程内直接 import:`list_tools()` 枚举工具,`call_tool(name, args)` 执行
     (CodingAgent 走这条路,免去进程间通信;自检也用它)

工具集(7 个):read_file / write_file / list_files / search_code /
run_tests / git_diff / git_apply —— 全部无状态,路径由调用方显式传入。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

_TOOL_IMPLS = {}
_TOOL_SCHEMAS = []

SKIP_DIRS = {".git", ".venv", "__pycache__", ".idea", "node_modules", ".pytest_cache"}
TEXT_EXTS = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
             ".cfg", ".ini", ".rst", ".html", ".js", ".ts", ".css"}


def register(name, description, properties, required):
    def deco(fn):
        _TOOL_IMPLS[name] = fn
        _TOOL_SCHEMAS.append({
            "name": name,
            "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": required},
        })
        return fn
    return deco


def list_tools():
    """MCP tools/list 的进程内等价物;自检据此枚举工具。"""
    return [dict(s) for s in _TOOL_SCHEMAS]


def call_tool(name, args):
    if name not in _TOOL_IMPLS:
        return f"错误:未知工具 {name!r},可用 {sorted(_TOOL_IMPLS)}"
    try:
        return str(_TOOL_IMPLS[name](args or {}))
    except Exception as e:
        return f"工具 {name} 执行出错: {type(e).__name__}: {e}"


# ----------------------------- 工具实现 -----------------------------

@register("read_file", "读取文本文件,返回带行号的内容",
          {"path": {"type": "string", "description": "文件路径(绝对或相对 cwd)"}},
          ["path"])
def _read_file(args):
    p = Path(args["path"])
    if not p.exists():
        return f"错误:文件不存在 {p}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = [f"{i:4d}| {ln}" for i, ln in enumerate(lines, 1)]
    out = "\n".join(numbered)
    return out[:8000] + ("\n...(截断)" if len(out) > 8000 else "")


@register("write_file", "整文件覆盖写入(自动建父目录)。修改代码时先 read_file 再写完整新内容",
          {"path": {"type": "string"},
           "content": {"type": "string", "description": "文件完整新内容"}},
          ["path", "content"])
def _write_file(args):
    p = Path(args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"], encoding="utf-8")
    return f"已写入 {p}({len(args['content'])} 字符)"


@register("list_files", "列出目录下的文件(递归,跳过 .git/.venv 等)",
          {"dir": {"type": "string", "description": "目录路径"}},
          ["dir"])
def _list_files(args):
    base = Path(args["dir"])
    if not base.is_dir():
        return f"错误:目录不存在 {base}"
    out = []
    for p in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            out.append(str(p.relative_to(base)))
        if len(out) >= 200:
            out.append("...(截断)")
            break
    return "\n".join(out) or "(空目录)"


@register("search_code", "在目录中用正则搜索代码,返回 文件:行号:内容",
          {"pattern": {"type": "string", "description": "正则表达式"},
           "dir": {"type": "string"}},
          ["pattern", "dir"])
def _search_code(args):
    base = Path(args["dir"])
    rx = re.compile(args["pattern"])
    hits = []
    for p in base.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            for i, ln in enumerate(
                    p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{p.relative_to(base)}:{i}: {ln.strip()[:120]}")
                if len(hits) >= 50:
                    return "\n".join(hits) + "\n...(截断)"
        except OSError:
            continue
    return "\n".join(hits) or f"未找到匹配 {args['pattern']!r}"


@register("run_tests", "在指定目录运行 python -m pytest,返回结果",
          {"dir": {"type": "string", "description": "仓库目录"},
           "extra_args": {"type": "string", "description": "附加 pytest 参数,如 -k add"}},
          ["dir"])
def _run_tests(args):
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if args.get("extra_args"):
        cmd += str(args["extra_args"]).split()
    try:
        r = subprocess.run(cmd, cwd=args["dir"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired:
        return "错误:pytest 超时(120s)"
    out = (r.stdout or "") + (r.stderr or "")
    return f"exit_code={r.returncode}\n{out[-2000:]}"


@register("git_diff", "显示 git 仓库的未提交改动(unified diff)",
          {"dir": {"type": "string"}}, ["dir"])
def _git_diff(args):
    r = subprocess.run(["git", "diff"], cwd=args["dir"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=30)
    if r.returncode != 0:
        return f"git diff 失败(不是 git 仓库?): {r.stderr.strip()[:200]}"
    return r.stdout[:8000] or "(无改动)"


@register("git_apply", "把 unified diff 补丁应用到 git 仓库",
          {"dir": {"type": "string"},
           "patch": {"type": "string", "description": "unified diff 文本"}},
          ["dir", "patch"])
def _git_apply(args):
    r = subprocess.run(["git", "apply", "-"], cwd=args["dir"], input=args["patch"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30)
    return "补丁已应用" if r.returncode == 0 else f"git apply 失败: {r.stderr.strip()[:300]}"


# ----------------------------- MCP stdio 协议 -----------------------------

def _handle(req):
    method = req.get("method", "")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05",
                  "serverInfo": {"name": "mini-coding-agent-mcp", "version": "0.1"},
                  "capabilities": {"tools": {}}}
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        params = req.get("params", {})
        text = call_tool(params.get("name"), params.get("arguments", {}))
        result = {"content": [{"type": "text", "text": text}]}
    elif method.startswith("notifications/"):
        return None                       # 通知无需响应
    else:
        return {"jsonrpc": "2.0", "id": req.get("id"),
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": req.get("id"), "result": result}


def main():
    """newline-delimited JSON-RPC over stdio(MCP stdio transport)。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(req)
        if resp is not None and req.get("id") is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
