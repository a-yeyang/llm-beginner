"""任务六消融:纯 prompt vs +Skill,单 agent vs +Subagent,在 toy-repo 上比步数与成败。

每个配置跑 2 次(LLM 有随机性),记录:是否修复成功、工具调用步数。
用法:python run_ablation.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent

TOY = ROOT / "data" / "toy-repo"
N_RUNS = 2

CONFIGS = [
    ("baseline(无skill无subagent)", dict(use_skills=False, use_subagents=False)),
    ("+Skill", dict(use_skills=True, use_subagents=False)),
    ("+Subagent", dict(use_skills=False, use_subagents=True)),
    ("+Skill+Subagent", dict(use_skills=True, use_subagents=True)),
]


def reset_repo():
    shutil.copy(TOY / "calculator.py.orig", TOY / "calculator.py")


def tests_pass():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=TOY,
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0


def main():
    issue = (TOY / "ISSUE.md").read_text(encoding="utf-8")
    results = []
    for name, kw in CONFIGS:
        for run_i in range(N_RUNS):
            reset_repo()
            agent = CodingAgent(**kw)
            trace = agent.run(repo_path=str(TOY), issue=issue)
            n_tool_steps = sum(1 for s in trace["steps"] if s.get("type") == "tool")
            ok = tests_pass()
            results.append({"config": name, "run": run_i + 1,
                            "success": ok, "tool_steps": n_tool_steps})
            print(f"{name} run{run_i + 1}: success={ok} tool_steps={n_tool_steps}")
    (ROOT / "eval" / "ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    reset_repo()   # 留下 buggy 版,便于他人复现自检
    print("\n结果写入 eval/ablation.json")


if __name__ == "__main__":
    main()
