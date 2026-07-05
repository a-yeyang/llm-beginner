"""Skill 加载器:扫描 skills/*/SKILL.md,按需把能力包内容注入 agent context。

Skill = 渐进式披露的能力包:平时 agent 只看到 name+description(便宜),
匹配到相关任务时才 load() 全文(贵)。SKILL.md 格式:
    ---
    name: test-runner
    description: 测试运行与失败诊断……
    ---
    (正文:workflow、注意事项、示例)
"""
import re
from pathlib import Path


class SkillLoader:
    def __init__(self, skills_dir):
        self.skills_dir = Path(skills_dir)
        self._skills = {}
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            meta = self._parse_frontmatter(md.read_text(encoding="utf-8"))
            name = meta.get("name") or md.parent.name
            self._skills[name] = {"name": name,
                                  "description": meta.get("description", ""),
                                  "path": str(md)}

    @staticmethod
    def _parse_frontmatter(text):
        m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        meta = {}
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        return meta

    def list_skills(self):
        """[{name, description, path}, ...] —— agent 的"技能目录"。"""
        return [dict(s) for s in self._skills.values()]

    def load(self, name):
        """按名加载 SKILL.md 全文(去掉 frontmatter)。"""
        if name not in self._skills:
            raise KeyError(f"未知 Skill: {name!r},可用 {sorted(self._skills)}")
        text = Path(self._skills[name]["path"]).read_text(encoding="utf-8")
        return re.sub(r"^\s*---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)

    def match(self, task):
        """朴素关键词匹配:description 与任务文本的词重叠得分,返回最相关 skill 名或 None。"""
        task_lower = task.lower()
        best, best_score = None, 0
        for name, s in self._skills.items():
            words = set(re.findall(r"[\w一-鿿]{2,}", s["description"].lower()))
            score = sum(1 for w in words if w in task_lower)
            if score > best_score:
                best, best_score = name, score
        return best
