"""测试执行 subagent:跑 pytest 并诊断失败原因,回传摘要。"""
from .base import BaseSubagent


class TesterSubagent(BaseSubagent):
    NAME = "tester"
    TOOL_NAMES = ("run_tests", "read_file")
    MAX_STEPS = 5
    SYSTEM = (
        "你是测试执行 subagent。在指定仓库运行 pytest,"
        "如有失败,读相关测试文件,输出诊断摘要:"
        "几个通过/失败、失败测试名、断言期望 vs 实际、你推断的根因(文件+函数)。"
        "只诊断,不要修改任何文件。"
    )
