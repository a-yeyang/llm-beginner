"""代码搜索 subagent:在仓库里定位与问题相关的文件/函数,回传摘要。"""
from .base import BaseSubagent


class ExplorerSubagent(BaseSubagent):
    NAME = "explorer"
    TOOL_NAMES = ("list_files", "read_file", "search_code")
    MAX_STEPS = 6
    SYSTEM = (
        "你是代码搜索 subagent。给定一个关于代码库的问题,"
        "用 list_files / search_code / read_file 找到答案,"
        "输出简洁摘要:相关文件路径、关键函数、关键行号与代码片段(不超过 15 行)。"
        "不要修改任何文件。"
    )
