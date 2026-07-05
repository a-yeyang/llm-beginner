"""下载 chinese-poetry 全唐诗语料，拼成纯文本 data/poems.txt。

数据源：github.com/chinese-poetry/chinese-poetry （繁体）。
直连 raw.githubusercontent，失败则走 ghproxy 镜像。
"""
import json
import sys
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent
N_FILES = 40  # 每个文件约 1000 首；40 个 ≈ 4 万首
BASE = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.{}.json"
MIRROR = "https://ghproxy.net/" + BASE


def fetch(idx):
    for url in (BASE.format(idx), MIRROR.format(idx)):
        for _ in range(2):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                time.sleep(0.5)
    return None


def main():
    poems, n_ok = [], 0
    for k in range(N_FILES):
        idx = k * 1000
        data = fetch(idx)
        if data is None:
            print(f"  [跳过] poet.tang.{idx}.json")
            continue
        for p in data:
            para = p.get("paragraphs", [])
            if para:
                poems.append("".join(para))
        n_ok += 1
        if k % 5 == 0:
            print(f"  已下载 {n_ok} 个文件，累计 {len(poems)} 首 ...")

    if not poems:
        sys.exit("[错误] 一首都没下到，检查网络/镜像")

    text = "\n".join(poems)
    out = DATA / "poems.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n完成：{len(poems)} 首 / {len(text)} 字 / {len(set(text))} 个不同字 -> {out}")


if __name__ == "__main__":
    main()
