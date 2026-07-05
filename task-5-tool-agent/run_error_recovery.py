"""错误恢复实验:把每题第 1 次工具调用的结果替换成假错误,统计 agent 恢复成功率。

通过线(README):恢复成功率 > 40%。
用法:python run_error_recovery.py
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent


def normalize(t):
    t = str(t).lower().replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", t)


def matches(answer, expected):
    ans = normalize(answer)
    for e in expected:
        if isinstance(e, list):
            if not any(normalize(k) in ans for k in e):
                return False
        elif normalize(e) not in ans:
            return False
    return True


def main():
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    ok = 0
    rows = []
    for t in tasks:
        agent = ReActAgent(inject_error_at=1, max_steps=10)
        try:
            trace = agent.run(t["task"])
            hit = matches(trace["final_answer"], t["expected_answer_contains"])
        except Exception as e:
            trace, hit = {"steps": []}, False
            print(f"  [{t['id']}] 异常: {e}")
        ok += int(hit)
        rows.append({"id": t["id"], "recovered": hit,
                     "n_steps": len(trace.get("steps", [])),
                     "answer_preview": str(trace.get("final_answer", ""))[:80]})
        print(f"  [{t['id']}] recovered={hit} steps={len(trace.get('steps', []))}")

    rate = ok / len(tasks)
    print(f"\n错误恢复成功率: {ok}/{len(tasks)} = {rate:.0%}(通过线 40%)")
    out = ROOT / "eval" / "error_recovery.json"
    out.write_text(json.dumps({"rate": rate, "details": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
