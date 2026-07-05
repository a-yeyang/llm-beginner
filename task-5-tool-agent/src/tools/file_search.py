"""本地文件检索工具:文件名匹配 + 文本内容检索,并附内容预览。"""
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在本地目录中检索文件:pattern 同时匹配文件名与文本内容,"
            "返回命中的文件路径、命中方式和内容开头预览。"
            "dir 支持相对路径(相对任务根目录),默认任务根目录。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "文件名片段(如 README.md、.md)或要查找的文本(如 TODO)"},
                "dir": {"type": "string", "description": "检索目录,默认任务根目录"},
            },
            "required": ["pattern"],
        },
    },
}

ROOT = Path(__file__).resolve().parents[2]      # task-5-tool-agent/
TEXT_EXTS = {".md", ".txt", ".py", ".json", ".jsonl", ".yaml", ".yml",
             ".toml", ".cfg", ".ini", ".tex", ".csv", ".html"}
SKIP_DIRS = {".venv", ".venv-wsl", "__pycache__", ".git", ".idea",
             "node_modules", "models", "ckpt", "cache"}
MAX_RESULTS = 20
PREVIEW_CHARS = 300


def _iter_files(base):
    for p in base.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


def run(args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return "错误:pattern 为空"
    raw_dir = str(args.get("dir", "") or "").strip()
    base = Path(raw_dir) if raw_dir else ROOT
    if not base.is_absolute():
        base = ROOT / base
    if not base.exists():
        return f"错误:目录不存在 {base}"

    hits = []
    for p in _iter_files(base):
        rel = p.relative_to(base)
        how = []
        if pattern.lower() in p.name.lower():
            how.append("文件名")
        preview = ""
        if p.suffix.lower() in TEXT_EXTS and p.stat().st_size < 1_000_000:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if pattern in content:
                    how.append("内容")
                if how:
                    preview = content[:PREVIEW_CHARS].strip()
            except OSError:
                pass
        if how:
            entry = f"- {rel}(命中:{'+'.join(how)})"
            if preview:
                entry += f"\n  内容开头:{preview!r}"
            hits.append(entry)
        if len(hits) >= MAX_RESULTS:
            break

    if not hits:
        return f"在 {base} 下未找到匹配 {pattern!r} 的文件"
    return f"在 {base} 下找到 {len(hits)} 个匹配:\n" + "\n".join(hits)
