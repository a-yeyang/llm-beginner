"""维基百科工具:MediaWiki 搜索 API + REST 摘要 API。

query 含中文字符时查 zh.wikipedia,否则查 en.wikipedia;
网络异常直接抛出(自检对网络工具按跳过处理)。
"""
import re
import urllib.parse

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科:先搜索最相关条目,再返回该条目的摘要。"
            "适合查人物、概念、事件的基本事实(生卒年、发明者、发表年份等)。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查询的条目或问题关键词"},
            },
            "required": ["query"],
        },
    },
}

_HEADERS = {"User-Agent": "llm-beginner-task5/1.0 (learning exercise)"}
_TIMEOUT = 12


def _lang_of(query):
    return "zh" if re.search(r"[一-鿿]", query) else "en"


def run(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return "错误:query 为空"
    lang = _lang_of(query)

    # 1) 全文搜索拿最相关标题
    r = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query,
                "srlimit": 3, "format": "json"},
        headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return f"维基百科({lang})未找到与 {query!r} 相关的条目"
    title = results[0]["title"]

    # 2) REST 摘要
    r2 = requests.get(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title, safe=""),
        headers=_HEADERS, timeout=_TIMEOUT)
    r2.raise_for_status()
    extract = r2.json().get("extract", "")

    other = ", ".join(x["title"] for x in results[1:])
    out = f"【{title}】{extract}"
    if other:
        out += f"\n(其他相关条目: {other})"
    return out[:1500]
