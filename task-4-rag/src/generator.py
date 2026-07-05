"""生成端:调 DashScope(阿里云百炼)OpenAI 兼容接口的 qwen-plus。

说明:README 建议本地 Qwen2.5-7B-Instruct,但 8GB 显存装不下 bf16 7B;
改用 API 生成不影响本任务的核心(检索质量与 prompt 组织),且生成端可插拔。
API Key 从仓库根 .env(DASHSCOPE_API_KEY)或环境变量读取,不写进代码。
"""
import os
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能依据给定的【资料】回答问题:\n"
    "1. 答案必须能从资料中找到依据,并在句末标注来源编号,如 [1][3]\n"
    "2. 资料不足以回答时,明确说「根据提供的资料无法回答」,不要编造\n"
    "3. 用中文简洁作答"
)


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


class Generator:
    def __init__(self, model=None):
        self.model = model or os.environ.get("RAG_GEN_MODEL", "qwen-plus")
        self.client = OpenAI(
            api_key=_load_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate(self, query, contexts, max_tokens=512, temperature=0.3):
        """contexts: List[dict(text, source)] -> 答案文本。"""
        ctx = "\n\n".join(
            f"[{i}] ({c.get('source', '?')}) {c['text']}"
            for i, c in enumerate(contexts, 1)
        )
        user = f"【资料】\n{ctx}\n\n【问题】{query}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
